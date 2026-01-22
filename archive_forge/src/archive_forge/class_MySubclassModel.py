import collections
import keras.src as keras
class MySubclassModel(keras.Model):
    """A subclass model."""

    def __init__(self, input_dim=3):
        super().__init__(name='my_subclass_model')
        self._config = {'input_dim': input_dim}
        self.dense1 = keras.layers.Dense(8, activation='relu')
        self.dense2 = keras.layers.Dense(2, activation='softmax')
        self.bn = keras.layers.BatchNormalization()
        self.dp = keras.layers.Dropout(0.5)

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dp(x)
        x = self.bn(x)
        return self.dense2(x)

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config):
        return cls(**config)