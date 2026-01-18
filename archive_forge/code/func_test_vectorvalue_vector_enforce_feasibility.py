import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def test_vectorvalue_vector_enforce_feasibility(self):
    m = 3
    lb = [1, 2, 3]
    ub = [4, 6, np.inf]
    enforce_feasibility = [True, False, True]
    strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
    assert_array_equal(strict_lb, [1, -np.inf, 3])
    assert_array_equal(strict_ub, [4, np.inf, np.inf])