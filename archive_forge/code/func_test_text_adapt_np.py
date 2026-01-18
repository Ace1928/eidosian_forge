import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import input_adapters
from autokeras.utils import data_utils
def test_text_adapt_np():
    x = np.array(['a b c', 'b b c'])
    adapter = input_adapters.TextAdapter()
    x = adapter.adapt(x, batch_size=32)
    assert data_utils.dataset_shape(x).as_list() == [None]
    assert isinstance(x, tf.data.Dataset)