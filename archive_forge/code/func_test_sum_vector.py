import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_sum_vector(self) -> None:
    w = cvxpy.Variable(2, pos=True)
    h = cvxpy.Variable(2, pos=True)
    problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(h)), [cvxpy.multiply(w, h) >= 10, cvxpy.sum(w) <= 10])
    problem.solve(SOLVER, gp=True)
    np.testing.assert_almost_equal(problem.value, 4)
    np.testing.assert_almost_equal(h.value, np.array([2, 2]))
    np.testing.assert_almost_equal(w.value, np.array([5, 5]))