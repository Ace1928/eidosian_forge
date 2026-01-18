from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
import cvxpy as cp
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest
def test_obj_eval(self) -> None:
    """Test case where objective evaluation differs from result.
        """
    x = cp.Variable((2, 1))
    A = np.array([[1.0]])
    B = np.array([[1.0, 1.0]]).T
    obj0 = -B.T @ x
    obj1 = cp.quad_form(B.T @ x, A)
    prob = cp.Problem(cp.Minimize(obj0 + obj1))
    prob.solve(solver=cp.SCS)
    self.assertAlmostEqual(prob.value, prob.objective.value)