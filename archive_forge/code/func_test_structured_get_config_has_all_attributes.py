import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import analysers
from autokeras import blocks
from autokeras import test_utils
def test_structured_get_config_has_all_attributes():
    block = blocks.StructuredDataBlock()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.StructuredDataBlock.__init__).issubset(config.keys())