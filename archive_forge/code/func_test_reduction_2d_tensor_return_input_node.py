import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_reduction_2d_tensor_return_input_node():
    block = blocks.TemporalReduction()
    input_node = keras.Input(shape=(32,), dtype=tf.float32)
    outputs = block.build(keras_tuner.HyperParameters(), input_node)
    assert len(nest.flatten(outputs)) == 1
    assert nest.flatten(outputs)[0] is input_node