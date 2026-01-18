import copy
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import input_analysers
def test_time_series_input_with_illegal_dim():
    analyser = input_analysers.TimeseriesAnalyser(column_names=test_utils.COLUMN_NAMES, column_types=None)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32)).batch(32)
    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the data to TimeseriesInput to have shape' in str(info.value)