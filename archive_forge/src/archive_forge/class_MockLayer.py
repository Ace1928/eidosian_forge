from unittest import mock
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import autokeras as ak
from autokeras import keras_layers
from autokeras import test_utils
from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
class MockLayer(preprocessing.TextVectorization):

    def adapt(self, *args, **kwargs):
        super().adapt(*args, **kwargs)
        self.is_called = True