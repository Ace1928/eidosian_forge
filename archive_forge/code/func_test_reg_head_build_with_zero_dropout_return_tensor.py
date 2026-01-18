import keras_tuner
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
import autokeras as ak
from autokeras import hyper_preprocessors
from autokeras import nodes as input_module
from autokeras import preprocessors
from autokeras import test_utils
from autokeras.blocks import heads as head_module
def test_reg_head_build_with_zero_dropout_return_tensor():
    block = head_module.RegressionHead(dropout=0, shape=(8,))
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(5,), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1