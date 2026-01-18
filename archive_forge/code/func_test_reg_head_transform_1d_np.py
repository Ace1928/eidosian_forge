import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import output_adapters
def test_reg_head_transform_1d_np():
    adapter = output_adapters.RegressionAdapter(name='a')
    y = adapter.adapt(np.random.rand(10), batch_size=32)
    assert isinstance(y, tf.data.Dataset)