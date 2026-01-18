import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def test_prepare_constraint_infeasible_x0():
    lb = np.array([0, 20, 30])
    ub = np.array([0.5, np.inf, 70])
    x0 = np.array([1, 2, 3])
    enforce_feasibility = np.array([False, True, True], dtype=bool)
    bounds = Bounds(lb, ub, enforce_feasibility)
    pytest.raises(ValueError, PreparedConstraint, bounds, x0)
    pc = PreparedConstraint(Bounds(lb, ub), [1, 2, 3])
    assert (pc.violation([1, 2, 3]) > 0).any()
    assert (pc.violation([0.25, 21, 31]) == 0).all()
    x0 = np.array([1, 2, 3, 4])
    A = np.array([[1, 2, 3, 4], [5, 0, 0, 6], [7, 0, 8, 0]])
    enforce_feasibility = np.array([True, True, True], dtype=bool)
    linear = LinearConstraint(A, -np.inf, 0, enforce_feasibility)
    pytest.raises(ValueError, PreparedConstraint, linear, x0)
    pc = PreparedConstraint(LinearConstraint(A, -np.inf, 0), [1, 2, 3, 4])
    assert (pc.violation([1, 2, 3, 4]) > 0).any()
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()

    def fun(x):
        return A.dot(x)

    def jac(x):
        return A

    def hess(x, v):
        return sps.csr_matrix((4, 4))
    nonlinear = NonlinearConstraint(fun, -np.inf, 0, jac, hess, enforce_feasibility)
    pytest.raises(ValueError, PreparedConstraint, nonlinear, x0)
    pc = PreparedConstraint(nonlinear, [-10, 2, -10, 4])
    assert (pc.violation([1, 2, 3, 4]) > 0).any()
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()