import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_three_constraints_16878():
    rng = np.random.default_rng(5123833489170494244)
    A = rng.integers(0, 5, size=(6, 6))
    bl = np.full(6, fill_value=-np.inf)
    bu = np.full(6, fill_value=10)
    constraints = [LinearConstraint(A[:2], bl[:2], bu[:2]), LinearConstraint(A[2:4], bl[2:4], bu[2:4]), LinearConstraint(A[4:], bl[4:], bu[4:])]
    constraints2 = [(A[:2], bl[:2], bu[:2]), (A[2:4], bl[2:4], bu[2:4]), (A[4:], bl[4:], bu[4:])]
    lb = np.zeros(6)
    ub = np.ones(6)
    variable_bounds = Bounds(lb, ub)
    c = -np.ones(6)
    res1 = milp(c, bounds=variable_bounds, constraints=constraints)
    res2 = milp(c, bounds=variable_bounds, constraints=constraints2)
    ref = milp(c, bounds=variable_bounds, constraints=(A, bl, bu))
    assert res1.success and res2.success
    assert_allclose(res1.x, ref.x)
    assert_allclose(res2.x, ref.x)