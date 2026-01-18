import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_solve_dpp_problem(self) -> None:
    x = cp.Parameter()
    x.value = 5
    y = cp.Variable()
    problem = cp.Problem(cp.Minimize(x + y), [x == y])
    self.assertTrue(problem.is_dpp())
    self.assertTrue(problem.is_dcp())
    self.assertAlmostEqual(problem.solve(cp.SCS), 10)
    x.value = 3
    self.assertAlmostEqual(problem.solve(cp.SCS), 6)