from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
import cvxpy as cp
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest
def test_singular_quad_form(self) -> None:
    """Test quad form with a singular matrix.
        """
    np.random.seed(1234)
    for n in (3, 4, 5):
        for i in range(5):
            v = np.exp(np.random.randn(n))
            v = v / np.sum(v)
            A = np.random.randn(n, n)
            Q = np.dot(A, A.T)
            E = np.identity(n) - np.outer(v, v) / np.inner(v, v)
            Q = np.dot(E, np.dot(Q, E.T))
            observed_rank = np.linalg.matrix_rank(Q)
            desired_rank = n - 1
            assert_equal(observed_rank, desired_rank)
            for action in ('minimize', 'maximize'):
                x = cp.Variable(n)
                if action == 'minimize':
                    q = cp.quad_form(x, Q)
                    objective = cp.Minimize(q)
                elif action == 'maximize':
                    q = cp.quad_form(x, -Q)
                    objective = cp.Maximize(q)
                constraints = [0 <= x, cp.sum(x) == 1]
                p = cp.Problem(objective, constraints)
                p.solve(solver=cp.OSQP)
                xopt = x.value.flatten()
                yopt = np.dot(xopt, np.dot(Q, xopt))
                assert_allclose(yopt, 0, atol=0.001)
                assert_allclose(xopt, v, atol=0.001)