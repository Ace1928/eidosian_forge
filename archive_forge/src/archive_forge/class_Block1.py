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
class Block1(ak.Block):

    def build(self, hp, inputs):
        hp.Boolean('a')
        return keras.layers.Dense(3)(tf.nest.flatten(inputs)[0])