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
def test_solve_problem_twice(self) -> None:
    """Test a problem with log.
        """
    n = 5
    x = cp.Variable(n)
    obj = cp.Maximize(cp.sum(cp.log(x)))
    p = cp.Problem(obj, [cp.sum(x) == 1])
    p.solve(solver=cp.SCS)
    first_value = x.value
    self.assertItemsAlmostEqual(first_value, n * [1.0 / n])
    p.solve(solver=cp.SCS)
    second_value = x.value
    self.assertItemsAlmostEqual(first_value, second_value)