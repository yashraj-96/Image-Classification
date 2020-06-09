# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:06:22 2020

@author: lenovo
"""

#Building the CNN 
#Importing the required packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.layers import Dropout

#Initializing the CNN
X = Sequential()

#Proceeding with the steps
#Convolution
X.add(Convolution2D(32,3 ,3 , input_shape=(128, 128, 3),activation='relu'))

#Pooling
X.add(MaxPooling2D(pool_size =(2,2)))

#Second convolutional layer / Pooling layer
X.add(Convolution2D(32,3,3 , activation='relu'))
X.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
X.add(Flatten())

#Fully-Connected layer
X.add(Dense(output_dim= 128, activation='relu'))
X.add(Dropout(p=0.1))
X.add(Dense(output_dim=1, activation ='sigmoid'))

#Compliling the CNN
X.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
X.fit(  train_set,  
        steps_per_epoch=8000,
        epochs=12,
        validation_data=test_set,
        validation_steps=2000)

#Making single predictions
import numpy as np
from keras.preprocessing import image
test_img=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(128, 128))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
output=X.predict(test_img)

#To get the indices corresponding to their respective classes
train_set.class_indices

if output[0][0] == 1:
    print('It is a dog')
else:
    print('It is a cat')




      
    















