from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
def test_reflective_transformation():
    lb = np.array([-1, -2], dtype=float)
    ub = np.array([5, 3], dtype=float)
    y = np.array([0, 0])
    x, g = reflective_transformation(y, lb, ub)
    assert_equal(x, y)
    assert_equal(g, np.ones(2))
    y = np.array([-4, 4], dtype=float)
    x, g = reflective_transformation(y, lb, np.array([np.inf, np.inf]))
    assert_equal(x, [2, 4])
    assert_equal(g, [-1, 1])
    x, g = reflective_transformation(y, np.array([-np.inf, -np.inf]), ub)
    assert_equal(x, [-4, 2])
    assert_equal(g, [1, -1])
    x, g = reflective_transformation(y, lb, ub)
    assert_equal(x, [2, 2])
    assert_equal(g, [-1, -1])
    lb = np.array([-np.inf, -2])
    ub = np.array([5, np.inf])
    y = np.array([10, 10], dtype=float)
    x, g = reflective_transformation(y, lb, ub)
    assert_equal(x, [0, 10])
    assert_equal(g, [-1, 1])