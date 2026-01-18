from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
import cvxpy as cp
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest
def test_nsd_exactly_tolerance(self) -> None:
    """Test that NSD check when eigenvalue is exactly EIGVAL_TOL
        """
    P = np.array([[0.999 * EIGVAL_TOL, 0], [0, -10]])
    x = cp.Variable(2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cost = cp.quad_form(x, P)
        prob = cp.Problem(cp.Maximize(cost), [x == [1, 2]])
        prob.solve(solver=cp.SCS)