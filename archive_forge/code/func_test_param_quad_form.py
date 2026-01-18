from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
import cvxpy as cp
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest
def test_param_quad_form(self) -> None:
    """Test quad form with a parameter.
        """
    P = cp.Parameter((2, 2), PSD=True)
    Q = np.eye(2)
    x = cp.Variable(2)
    cost = cp.quad_form(x, P)
    P.value = Q
    prob = cp.Problem(cp.Minimize(cost), [x == [1, 2]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        self.assertAlmostEqual(prob.solve(solver=cp.SCS), 5)