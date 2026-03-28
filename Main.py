from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from ObjectDetection import detectObject, displayImage
from keras.models import Sequential, load_model, Model
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import os


main = tkinter.Tk()
main.title("Smart Artificial Intelligence Based Online Proctoring System")
main.geometry("1300x1200")

class_labels = open('model/model-labels').read().strip().split('\n') #reading labels from model
cnn_model = cv2.dnn.readNetFromDarknet('model/model.cfg', 'model/model.weights') #reading model
cnn_layer_names = cnn_model.getLayerNames() #getting layers from cnn model
for i in cnn_model.getUnconnectedOutLayers():
    print(i)
cnn_layer_names = [cnn_layer_names[i - 1] for i in cnn_model.getUnconnectedOutLayers()] #assigning all layers
label_colors = (0, 0, 255)
global emotion_model

faceCascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

def loadModel():
    global emotion_model
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            emotion_model = model_from_json(loaded_model_json)
        json_file.close()    
        emotion_model.load_weights("model/cnnmodel_weights.h5")
        emotion_model._make_predict_function()                  
    else:
        emotion_model = Sequential()
        emotion_model.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
        emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
        emotion_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
        emotion_model.add(Flatten())
        emotion_model.add(Dense(output_dim = 256, activation = 'relu'))
        emotion_model.add(Dense(output_dim = 7, activation = 'softmax'))
        emotion_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = emotion_model.fit(image_X_train, image_y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        emotion_model.save_weights('model/cnnmodel_weights.h5')            
        model_json = emotion_model.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/cnnhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = emotion_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    a = accuracy_score(y_test1,predict)*100
    p = precision_score(y_test1, predict,average='macro') * 100
    r = recall_score(y_test1, predict,average='macro') * 100
    f = f1_score(y_test1, predict,average='macro') * 100
    algorithm = "CNN"
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

def detectEmotion(image):
    global emotion_model
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    predict = 4
    for (x, y, w, h) in faces:
        img = image[y:y+h, x:x+w]
        #cv2.imwrite("face.jpg", img)    
        img = cv2.resize(img, (32,32))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = emotion_model.predict(img)
        predict = np.argmax(preds)        
    return labels[predict]

def getPose(image):
    pose = ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        center_x = (x + x + w) // 2
        center_y = (y + y + h) // 2
        distance_x = center_x - image.shape[1] // 2
        distance_y = center_y - image.shape[0] // 2
        if distance_x > 0:
            pose = "Right"
        elif distance_x < 0:
            pose = "Left"
    return pose        


def webcamVideo():
    text.delete('1.0', END)
    webcamera = cv2.VideoCapture(0)

    while True:
        (grab, frame) = webcamera.read()
        if not grab:
            break

        frame_height, frame_width = frame.shape[:2]

        frames, cls, Boundingboxes, confidence_value, class_ids, ids = detectObject(
            cnn_model, cnn_layer_names,
            frame_height, frame_width,
            frame, label_colors, class_labels
        )

        if ids is not None and len(ids) > 0:
            ids = np.array(ids).flatten()

            for i in ids:
                xx, yy = Boundingboxes[i][0], Boundingboxes[i][1]
                width, height = Boundingboxes[i][2], Boundingboxes[i][3]

                if class_ids[i] == 0:
                    pose = getPose(frame)
                    emotion = detectEmotion(frame)

                    cv2.putText(frames, emotion, (xx, yy-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    cv2.putText(frames, "Head Pose: " + pose, (xx, yy-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Detected Objects", frames)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break

    webcamera.release()
    cv2.destroyAllWindows()
 
	


def exit():
    main.destroy()

    
font = ('times', 16, 'bold')
title = Label(main, text='Smart Artificial Intelligence Based Online Proctoring System')
title.config(bg='light cyan', fg='pale violet red')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Generate & Load CNN Model", command=loadModel)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='light cyan', fg='pale violet red')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

webcamButton = Button(main, text="Webcam Based Proctoring System", command=webcamVideo)
webcamButton.place(x=50,y=150)
webcamButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=460,y=150)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
