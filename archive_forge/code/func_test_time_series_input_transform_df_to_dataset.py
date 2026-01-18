import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import input_adapters
from autokeras.utils import data_utils
def test_time_series_input_transform_df_to_dataset():
    adapter = input_adapters.TimeseriesAdapter()
    x = adapter.adapt(pd.DataFrame(np.random.rand(100, 32)), batch_size=32)
    assert isinstance(x, tf.data.Dataset)