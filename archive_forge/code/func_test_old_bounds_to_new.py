import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def test_old_bounds_to_new():
    bounds = ([1, 2], (None, 3), (-1, None))
    lb_true = np.array([1, -np.inf, -1])
    ub_true = np.array([2, 3, np.inf])
    lb, ub = old_bound_to_new(bounds)
    assert_array_equal(lb, lb_true)
    assert_array_equal(ub, ub_true)
    bounds = [(-np.inf, np.inf), (np.array([1]), np.array([1]))]
    lb, ub = old_bound_to_new(bounds)
    assert_array_equal(lb, [-np.inf, 1])
    assert_array_equal(ub, [np.inf, 1])