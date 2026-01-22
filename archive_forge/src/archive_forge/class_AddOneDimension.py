import tensorflow as tf
from autokeras import analysers
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import preprocessor
from autokeras.utils import data_utils
class AddOneDimension(LambdaPreprocessor):
    """Append one dimension of size one to the dataset shape."""

    def __init__(self, **kwargs):
        super().__init__(lambda x: tf.expand_dims(x, axis=-1), **kwargs)