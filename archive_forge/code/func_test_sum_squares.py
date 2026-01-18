import warnings
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_sum_squares(self) -> None:
    X = Variable((5, 4))
    P = np.ones((3, 5))
    Q = np.ones((4, 7))
    M = np.ones((3, 7))
    y = P @ X @ Q + M
    self.assertFalse(y.is_constant())
    self.assertTrue(y.is_affine())
    self.assertTrue(y.is_quadratic())
    self.assertTrue(y.is_dcp())
    s = cp.sum_squares(y)
    self.assertFalse(s.is_constant())
    self.assertFalse(s.is_affine())
    self.assertTrue(s.is_quadratic())
    self.assertTrue(s.is_dcp())
    t = cp.norm(y, 'fro') ** 2
    self.assertFalse(t.is_constant())
    self.assertFalse(t.is_affine())
    self.assertFalse(t.is_quadratic())
    self.assertTrue(t.is_dcp())