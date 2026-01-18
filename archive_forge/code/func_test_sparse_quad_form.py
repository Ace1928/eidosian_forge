from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
import cvxpy as cp
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest
def test_sparse_quad_form(self) -> None:
    """Test quad form with a sparse matrix.
        """
    Q = sp.eye(2)
    x = cp.Variable(2)
    cost = cp.quad_form(x, Q)
    prob = cp.Problem(cp.Minimize(cost), [x == [1, 2]])
    self.assertAlmostEqual(prob.solve(solver=cp.OSQP), 5)
    A = cp.Constant(sp.eye(4))
    c = np.ones(4).reshape((1, 4))
    x = cp.Variable(4)
    function = cp.quad_form(x, A) - cp.matmul(c, x)
    objective = cp.Minimize(function)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.OSQP)
    self.assertEqual(len(function.value), 1)