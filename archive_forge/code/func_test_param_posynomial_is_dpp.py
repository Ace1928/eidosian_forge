import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_param_posynomial_is_dpp(self) -> None:
    alpha = cp.Parameter(pos=True)
    beta = cp.Parameter(pos=True)
    kappa = cp.Parameter(pos=True)
    monomial = alpha ** 1.2 * beta ** 0.5 * kappa ** 3 * kappa ** 2
    posynomial = monomial + alpha ** 2 * beta ** 3
    self.assertTrue(posynomial.is_dgp(dpp=True))