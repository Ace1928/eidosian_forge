import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_with_bound(self):
    x0 = np.array([0.0, 0.85, -0.85])
    lb = -np.ones(3)
    ub = np.ones(3)
    h = np.array([1, 1, -1]) * 0.1
    h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
    assert_allclose(h_adjusted, h)
    h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
    assert_allclose(h_adjusted, np.array([1, -1, 1]) * 0.1)
    h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 1, '2-sided', lb, ub)
    assert_allclose(h_adjusted, np.abs(h))
    assert_(np.all(~one_sided))
    h_adjusted, one_sided = _adjust_scheme_to_bounds(x0, h, 2, '2-sided', lb, ub)
    assert_allclose(h_adjusted, np.array([1, -1, 1]) * 0.1)
    assert_equal(one_sided, np.array([False, True, True]))