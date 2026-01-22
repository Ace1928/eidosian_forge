import keras.src as keras
from keras.src.testing_infra import test_utils
class Inner(keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(5, activation='relu')
        self.bn = keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.bn(x)