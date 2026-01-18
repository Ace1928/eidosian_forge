import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_solve_multiply_param_plus_var_times_const(self) -> None:
    x = cp.Parameter()
    y = cp.Variable()
    product = (x + y) * 5
    self.assertTrue(product.is_dpp())
    x.value = 2.0
    problem = cp.Problem(cp.Minimize(product), [y == 1])
    value = problem.solve(cp.SCS)
    self.assertAlmostEqual(value, 15)