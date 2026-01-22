import keras.src as keras
from keras.src.testing_infra import test_utils
class SimpleConvTestModel(keras.Model):

    def __init__(self, num_classes=10):
        super().__init__(name='test_model')
        self.num_classes = num_classes
        self.conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        return self.dense1(x)