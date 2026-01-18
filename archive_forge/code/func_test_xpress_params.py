import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_xpress_params(self) -> None:
    if cp.XPRESS in INSTALLED_SOLVERS:
        n, m = (10, 4)
        np.random.seed(0)
        A = np.random.randn(m, n)
        x = np.random.randn(n)
        y = A.dot(x)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.norm1(z))
        constraints = [A @ z == y]
        problem = cp.Problem(objective, constraints)
        params = {'lpiterlimit': 1000, 'maxtime': 1000}
        problem.solve(solver=cp.XPRESS, **params)