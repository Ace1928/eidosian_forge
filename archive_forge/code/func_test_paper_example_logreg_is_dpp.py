import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_paper_example_logreg_is_dpp(self) -> None:
    N, n = (3, 2)
    beta = cp.Variable((n, 1))
    b = cp.Variable((1, 1))
    X = cp.Parameter((N, n))
    Y = np.ones((N, 1))
    lambd1 = cp.Parameter(nonneg=True)
    lambd2 = cp.Parameter(nonneg=True)
    log_likelihood = 1.0 / N * cp.sum(cp.multiply(Y, X @ beta + b) - cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), X @ beta + b]).T, axis=0, keepdims=True).T)
    regularization = -lambd1 * cp.norm(beta, 1) - lambd2 * cp.sum_squares(beta)
    problem = cp.Problem(cp.Maximize(log_likelihood + regularization))
    self.assertTrue(log_likelihood.is_dpp())
    self.assertTrue(problem.is_dcp())
    self.assertTrue(problem.is_dpp())