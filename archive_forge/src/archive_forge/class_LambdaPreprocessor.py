import tensorflow as tf
from autokeras import analysers
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import preprocessor
from autokeras.utils import data_utils
class LambdaPreprocessor(preprocessor.Preprocessor):
    """Build Preprocessor with a map function.

    # Arguments
        func: a callable function for the dataset to map.
    """

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def transform(self, dataset):
        return dataset.map(self.func)