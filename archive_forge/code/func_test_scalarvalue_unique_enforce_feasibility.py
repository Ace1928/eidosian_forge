import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def test_scalarvalue_unique_enforce_feasibility(self):
    m = 3
    lb = 2
    ub = 4
    enforce_feasibility = False
    strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
    assert_array_equal(strict_lb, [-np.inf, -np.inf, -np.inf])
    assert_array_equal(strict_ub, [np.inf, np.inf, np.inf])
    enforce_feasibility = True
    strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
    assert_array_equal(strict_lb, [2, 2, 2])
    assert_array_equal(strict_ub, [4, 4, 4])