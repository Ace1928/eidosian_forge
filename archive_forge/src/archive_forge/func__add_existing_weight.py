import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
def _add_existing_weight(self, weight, trainable):
    """Calls add_weight() to register but not create an existing weight."""
    self.add_weight(name=weight.name, shape=weight.shape, dtype=weight.dtype, trainable=trainable, getter=lambda *_, **__: weight)