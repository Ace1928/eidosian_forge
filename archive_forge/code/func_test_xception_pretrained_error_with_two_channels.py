import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_xception_pretrained_error_with_two_channels():
    block = blocks.XceptionBlock(pretrained=True)
    with pytest.raises(ValueError) as info:
        block.build(keras_tuner.HyperParameters(), keras.Input(shape=(224, 224, 2), dtype=tf.float32))
    assert 'When pretrained is set to True' in str(info.value)