import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_nonlinear_equality_not_dpp(self) -> None:
    x = cp.Variable()
    a = cp.Parameter()
    constraint = [x == cp.norm(a)]
    self.assertFalse(constraint[0].is_dcp(dpp=True))
    problem = cp.Problem(cp.Minimize(0), constraint)
    self.assertFalse(problem.is_dcp(dpp=True))