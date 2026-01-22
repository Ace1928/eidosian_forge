import tensorflow as tf
from autokeras import analysers
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import preprocessor
from autokeras.utils import data_utils
class SlidingWindow(preprocessor.Preprocessor):
    """Apply sliding window to the dataset.

    It groups the consecutive data items together. Therefore, it inserts one
    more dimension of size `lookback` to the dataset shape after the batch_size
    dimension. It also reduce the number of instances in the dataset by
    (lookback - 1).

    # Arguments
        lookback: Int. The window size. The number of data items to group
            together.
        batch_size: Int. The batch size of the dataset.
    """

    def __init__(self, lookback, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.batch_size = batch_size

    def transform(self, dataset):
        dataset = dataset.unbatch()
        dataset = dataset.window(self.lookback, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x: x.batch(self.lookback, drop_remainder=True))
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_config(self):
        return {'lookback': self.lookback, 'batch_size': self.batch_size}