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
def test_tuner_does_not_crash_with_distribution_strategy(tmp_path):
    tuner = greedy.Greedy(hypermodel=test_utils.build_graph(), directory=tmp_path, distribution_strategy=tf.distribute.MirroredStrategy())
    tuner.hypermodel.build(tuner.oracle.hyperparameters)