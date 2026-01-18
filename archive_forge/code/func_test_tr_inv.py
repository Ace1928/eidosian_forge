import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_tr_inv(self) -> None:
    """Test tr_inv atom. """
    T = 5
    X = cp.Variable((T, T), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.trace(X) == 1]
    prob = cp.Problem(cp.Minimize(cp.tr_inv(X)), constraints)
    prob.solve()
    self.assertAlmostEqual(prob.value, T ** 2)
    X_actual = X.value
    X_expect = np.eye(T) / T
    self.assertItemsAlmostEqual(X_actual, X_expect, places=4)
    constraints = [X >> 0]
    n = 4
    M = np.random.randn(n, T)
    constraints += [X >= -1, X <= 1]
    prob = cp.Problem(cp.Minimize(cp.tr_inv(M @ X @ M.T)), constraints)
    MM = M @ M.T
    naiveRes = np.sum(LA.eigvalsh(MM) ** (-1))
    prob.solve(verbose=True)
    self.assertTrue(prob.value < naiveRes)