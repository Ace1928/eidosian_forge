import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_rnn_input_shape_one_dim_error():
    block = blocks.RNNBlock()
    with pytest.raises(ValueError) as info:
        block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32,), dtype=tf.float32))
    assert 'Expect the input tensor of RNNBlock' in str(info.value)