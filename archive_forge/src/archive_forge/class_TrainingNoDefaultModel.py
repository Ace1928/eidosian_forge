import keras.src as keras
from keras.src.testing_infra import test_utils
class TrainingNoDefaultModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(1)

    def call(self, x, training):
        return self.dense1(x)