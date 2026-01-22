import keras.src as keras
from keras.src.testing_infra import test_utils
class NestedTestModel1(keras.Model):
    """A model subclass nested inside a model subclass."""

    def __init__(self, num_classes=2):
        super().__init__(name='nested_model_1')
        self.num_classes = num_classes
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='relu')
        self.bn = keras.layers.BatchNormalization()
        self.test_net = test_utils.SmallSubclassMLP(num_hidden=32, num_classes=4, use_bn=True, use_dp=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn(x)
        x = self.test_net(x)
        return self.dense2(x)