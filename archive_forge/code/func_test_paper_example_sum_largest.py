import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_paper_example_sum_largest(self) -> None:
    self.skipTest('Enable test once sum_largest is implemented.')
    x = cvxpy.Variable((4,), pos=True)
    x0, x1, x2, x3 = (x[0], x[1], x[2], x[3])
    obj = cvxpy.Minimize(cvxpy.sum_largest(cvxpy.hstack([3 * x0 ** 0.5 * x1 ** 0.5, x0 * x1 + 0.5 * x1 * x3 ** 3, x2]), 2))
    constr = [x0 * x1 * x2 >= 16]
    p = cvxpy.Problem(obj, constr)
    p.solve(SOLVER, gp=True)