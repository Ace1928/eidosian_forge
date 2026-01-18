import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def test_scalarvalue_vector_enforce_feasibility(self):
    m = 3
    lb = 2
    ub = 4
    enforce_feasibility = [False, True, False]
    strict_lb, strict_ub = strict_bounds(lb, ub, enforce_feasibility, m)
    assert_array_equal(strict_lb, [-np.inf, 2, -np.inf])
    assert_array_equal(strict_ub, [np.inf, 4, np.inf])