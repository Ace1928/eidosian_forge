import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_temporal_deserialize_to_temporal():
    serialized_block = blocks.serialize(blocks.TemporalReduction())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.TemporalReduction)