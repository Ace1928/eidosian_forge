import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy.optimize._pava_pybind import pava
from scipy.optimize import isotonic_regression
@pytest.mark.parametrize('y_dtype', [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize('w_dtype', [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize('w', [None, 'ones'])
def test_simple_isotonic_regression(self, w, w_dtype, y_dtype):
    y = np.array([8, 4, 8, 2, 2, 0, 8], dtype=y_dtype)
    if w is not None:
        w = np.ones_like(y, dtype=w_dtype)
    res = isotonic_regression(y, weights=w)
    assert res.x.dtype == np.float64
    assert res.weights.dtype == np.float64
    assert_allclose(res.x, [4, 4, 4, 4, 4, 4, 8])
    assert_allclose(res.weights, [6, 1])
    assert_allclose(res.blocks, [0, 6, 7])
    assert_equal(y, np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64))