import warnings
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_quadratic_form(self) -> None:
    x = Variable(5)
    P = np.eye(5) - 2 * np.ones((5, 5))
    q = np.ones((5, 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = x.T @ P @ x + q.T @ x
    self.assertFalse(s.is_constant())
    self.assertFalse(s.is_affine())
    self.assertTrue(s.is_quadratic())
    self.assertFalse(s.is_dcp())