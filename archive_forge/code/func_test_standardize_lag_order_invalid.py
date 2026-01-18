import numpy as np
from numpy.testing import assert_equal, assert_raises
from statsmodels.tsa.arima.tools import (
def test_standardize_lag_order_invalid():
    assert_raises(TypeError, standardize_lag_order, None)
    assert_raises(ValueError, standardize_lag_order, 1.2)
    assert_raises(ValueError, standardize_lag_order, -1)
    assert_raises(ValueError, standardize_lag_order, np.arange(4).reshape(2, 2))
    assert_raises(ValueError, standardize_lag_order, [0, 2])
    assert_raises(ValueError, standardize_lag_order, [1, 1, 2])