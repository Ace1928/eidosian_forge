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
def test_xpress_warm_start(self) -> None:
    """Make sure that warm starting Xpress behaves as expected
           Note: Xpress does not have warmstart yet, it will re-solve problem from scratch
        """
    if cp.XPRESS in INSTALLED_SOLVERS:
        import numpy as np
        A = cp.Parameter((2, 2))
        b = cp.Parameter(2)
        h = cp.Parameter(2)
        c = cp.Parameter(2)
        A.value = np.array([[1, 0], [0, 0]])
        b.value = np.array([1, 0])
        h.value = np.array([2, 2])
        c.value = np.array([1, 1])
        objective = cp.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
        constraints = [self.x[0] <= h[0], self.x[1] <= h[1], A @ self.x == b]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.XPRESS, warm_start=True)
        self.assertEqual(result, 3)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])
        A.value = np.array([[0, 0], [0, 1]])
        b.value = np.array([0, 1])
        h.value = np.array([2, 2])
        c.value = np.array([1, 1])
        result = prob.solve(solver=cp.XPRESS, warm_start=True)
        self.assertEqual(result, 3)
        self.assertItemsAlmostEqual(self.x.value, [2, 1])
        A.value = np.array([[1, 0], [0, 0]])
        b.value = np.array([1, 0])
        h.value = np.array([1, 1])
        c.value = np.array([1, 1])
        result = prob.solve(solver=cp.XPRESS, warm_start=True)
        self.assertEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])
        A.value = np.array([[1, 0], [0, 0]])
        b.value = np.array([1, 0])
        h.value = np.array([2, 2])
        c.value = np.array([2, 1])
        result = prob.solve(solver=cp.XPRESS, warm_start=True)
        self.assertEqual(result, 4)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])
    else:
        with self.assertRaises(Exception) as cm:
            prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
            prob.solve(solver=cp.XPRESS, warm_start=True)
        self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % cp.XPRESS)