import numpy as np
import tensorflow as tf
from autokeras import test_utils
from autokeras.preprocessors import common
from autokeras.utils import data_utils
def test_time_series_input_transform():
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32)).batch(32)
    preprocessor = common.SlidingWindow(lookback=2, batch_size=32)
    x = preprocessor.transform(dataset)
    assert data_utils.dataset_shape(x).as_list() == [None, 2, 32]