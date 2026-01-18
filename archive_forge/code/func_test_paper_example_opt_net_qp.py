import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_paper_example_opt_net_qp(self) -> None:
    m, n = (3, 2)
    G = cp.Parameter((m, n))
    h = cp.Parameter((m, 1))
    p = cp.Parameter((n, 1))
    y = cp.Variable((n, 1))
    objective = cp.Minimize(0.5 * cp.sum_squares(y - p))
    constraints = [G @ y <= h]
    problem = cp.Problem(objective, constraints)
    self.assertTrue(problem.is_dpp())