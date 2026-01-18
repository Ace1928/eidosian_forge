import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import analysers
from autokeras import blocks
from autokeras import test_utils
def test_text_deserialize_to_text():
    serialized_block = blocks.serialize(blocks.TextBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.TextBlock)