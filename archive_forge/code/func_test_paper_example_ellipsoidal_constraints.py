import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_paper_example_ellipsoidal_constraints(self) -> None:
    n = 2
    A_sqrt = cp.Parameter((n, n))
    z = cp.Parameter(n)
    p = cp.Parameter(n)
    y = cp.Variable(n)
    slack = cp.Variable(y.shape)
    objective = cp.Minimize(0.5 * cp.sum_squares(y - p))
    constraints = [0.5 * cp.sum_squares(A_sqrt @ slack) <= 1, slack == y - z]
    problem = cp.Problem(objective, constraints)
    self.assertTrue(problem.is_dpp())