import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import output_adapters
def test_reg_head_transform_pd_series():
    adapter = output_adapters.RegressionAdapter(name='a')
    y = adapter.adapt(pd.read_csv(test_utils.TEST_CSV_PATH).pop('survived'), batch_size=32)
    assert isinstance(y, tf.data.Dataset)