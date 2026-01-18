from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
def test_scaling_vector(self):
    lb = np.array([-np.inf, -5.0, 1.0, -np.inf])
    ub = np.array([1.0, np.inf, 10.0, np.inf])
    x = np.array([0.5, 2.0, 5.0, 0.0])
    g = np.array([1.0, 0.1, -10.0, 0.0])
    v, dv = CL_scaling_vector(x, g, lb, ub)
    assert_equal(v, [1.0, 7.0, 5.0, 1.0])
    assert_equal(dv, [0.0, 1.0, -1.0, 0.0])