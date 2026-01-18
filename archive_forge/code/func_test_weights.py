import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy.optimize._pava_pybind import pava
from scipy.optimize import isotonic_regression
def test_weights(self):
    w = np.array([1, 2, 5, 0.5, 0.5, 0.5, 1, 3])
    y = np.array([3, 2, 1, 10, 9, 8, 20, 10])
    res = isotonic_regression(y, weights=w)
    assert_allclose(res.x, [12 / 8, 12 / 8, 12 / 8, 9, 9, 9, 50 / 4, 50 / 4])
    assert_allclose(res.weights, [8, 1.5, 4])
    assert_allclose(res.blocks, [0, 3, 6, 8])
    w2 = np.array([1, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1, 3])
    y2 = np.array([3, 2, 1, 1, 1, 1, 1, 10, 9, 8, 20, 10])
    res2 = isotonic_regression(y2, weights=w2)
    assert_allclose(np.diff(res2.x[0:7]), 0)
    assert_allclose(res2.x[4:], res.x)
    assert_allclose(res2.weights, res.weights)
    assert_allclose(res2.blocks[1:] - 4, res.blocks[1:])