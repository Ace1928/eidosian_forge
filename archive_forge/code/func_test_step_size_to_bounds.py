from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
def test_step_size_to_bounds(self):
    lb = np.array([-1.0, 2.5, 10.0])
    ub = np.array([1.0, 5.0, 100.0])
    x = np.array([0.0, 2.5, 12.0])
    s = np.array([0.1, 0.0, 0.0])
    step, hits = step_size_to_bound(x, s, lb, ub)
    assert_equal(step, 10)
    assert_equal(hits, [1, 0, 0])
    s = np.array([0.01, 0.05, -1.0])
    step, hits = step_size_to_bound(x, s, lb, ub)
    assert_equal(step, 2)
    assert_equal(hits, [0, 0, -1])
    s = np.array([10.0, -0.0001, 100.0])
    step, hits = step_size_to_bound(x, s, lb, ub)
    assert_equal(step, np.array(-0))
    assert_equal(hits, [0, -1, 0])
    s = np.array([1.0, 0.5, -2.0])
    step, hits = step_size_to_bound(x, s, lb, ub)
    assert_equal(step, 1.0)
    assert_equal(hits, [1, 0, -1])
    s = np.zeros(3)
    step, hits = step_size_to_bound(x, s, lb, ub)
    assert_equal(step, np.inf)
    assert_equal(hits, [0, 0, 0])