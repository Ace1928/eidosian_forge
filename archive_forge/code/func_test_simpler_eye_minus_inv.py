import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_simpler_eye_minus_inv(self) -> None:
    X = cvxpy.Variable((2, 2), pos=True)
    obj = cvxpy.Minimize(cvxpy.trace(cvxpy.eye_minus_inv(X)))
    constr = [cvxpy.diag(X) == 0.1, cvxpy.hstack([X[0, 1], X[1, 0]]) == 0.1]
    problem = cvxpy.Problem(obj, constr)
    problem.solve(gp=True, solver='ECOS')
    np.testing.assert_almost_equal(X.value, 0.1 * np.ones((2, 2)), decimal=3)
    self.assertAlmostEqual(problem.value, 2.25)