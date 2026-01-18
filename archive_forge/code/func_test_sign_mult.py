import unittest
from cvxpy import Constant, Variable, log
from cvxpy.settings import QUASILINEAR, UNKNOWN
def test_sign_mult(self) -> None:
    self.assertEqual((self.zero * self.cvx).curvature, self.aff.curvature)
    self.assertEqual((self.neg * self.cvx).curvature, self.ccv.curvature)
    self.assertEqual((self.neg * self.ccv).curvature, self.cvx.curvature)
    self.assertEqual((self.neg * self.unknown_curv).curvature, QUASILINEAR)
    self.assertEqual((self.pos * self.aff).curvature, self.aff.curvature)
    self.assertEqual((self.pos * self.ccv).curvature, self.ccv.curvature)
    self.assertEqual((self.unknown_sign * self.const).curvature, self.const.curvature)
    self.assertEqual((self.unknown_sign * self.ccv).curvature, UNKNOWN)