import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_param_in_exponent_and_elsewhere(self) -> None:
    alpha = cp.Parameter(pos=True, value=1.0, name='alpha')
    x = cp.Variable(pos=True)
    problem = cp.Problem(cp.Minimize(x ** alpha), [x == alpha])
    self.assertTrue(problem.is_dgp(dpp=True))
    problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
    self.assertAlmostEqual(problem.value, 1.0)
    self.assertAlmostEqual(x.value, 1.0)
    problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
    self.assertAlmostEqual(problem.value, 1.0)
    self.assertAlmostEqual(x.value, 1.0)
    alpha.value = 3.0
    problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
    self.assertAlmostEqual(problem.value, 27.0)
    self.assertAlmostEqual(x.value, 3.0)