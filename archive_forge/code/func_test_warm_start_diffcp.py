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
def test_warm_start_diffcp(self) -> None:
    """Test warm starting in diffcvx.
        """
    try:
        import diffcp
        diffcp
    except ModuleNotFoundError:
        self.skipTest('diffcp not installed.')
    x = cp.Variable(10)
    obj = cp.Minimize(cp.sum(cp.exp(x)))
    prob = cp.Problem(obj, [cp.sum(x) == 1])
    result = prob.solve(solver=cp.DIFFCP)
    result2 = prob.solve(solver=cp.DIFFCP, warm_start=True)
    self.assertAlmostEqual(result2, result, places=2)