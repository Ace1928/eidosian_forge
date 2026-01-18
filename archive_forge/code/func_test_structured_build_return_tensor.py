import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import analysers
from autokeras import blocks
from autokeras import test_utils
def test_structured_build_return_tensor():
    block = blocks.StructuredDataBlock()
    block.column_names = ['0', '1']
    block.column_types = {'0': analysers.NUMERICAL, '1': analysers.NUMERICAL}
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(2,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1