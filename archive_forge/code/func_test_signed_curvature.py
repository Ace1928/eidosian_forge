import cvxpy as cvx
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_signed_curvature(self) -> None:
    expr = cvx.abs(1 + cvx.exp(cvx.Variable()))
    self.assertEqual(expr.curvature, s.CONVEX)
    expr = cvx.abs(-cvx.entr(cvx.Variable()))
    self.assertEqual(expr.curvature, s.UNKNOWN)
    expr = cvx.abs(-cvx.log(cvx.Variable()))
    self.assertEqual(expr.curvature, s.UNKNOWN)
    expr = cvx.abs(cvx.log(cvx.Variable()))
    self.assertEqual(expr.curvature, s.UNKNOWN)
    expr = cvx.abs(-cvx.square(cvx.Variable()))
    self.assertEqual(expr.curvature, s.CONVEX)
    expr = cvx.abs(cvx.entr(cvx.Variable()))
    self.assertEqual(expr.curvature, s.UNKNOWN)
    expr = cvx.abs(cvx.Variable(nonneg=True))
    self.assertEqual(expr.curvature, s.CONVEX)
    expr = cvx.abs(-cvx.Variable(nonneg=True))
    self.assertEqual(expr.curvature, s.CONVEX)
    expr = cvx.abs(cvx.Variable())
    self.assertEqual(expr.curvature, s.CONVEX)