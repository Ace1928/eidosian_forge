import tensorflow as tf
from autokeras import analysers
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import preprocessor
from autokeras.utils import data_utils
class CastToString(preprocessor.Preprocessor):
    """Cast the dataset shape to tf.string."""

    def transform(self, dataset):
        return dataset.map(data_utils.cast_to_string)