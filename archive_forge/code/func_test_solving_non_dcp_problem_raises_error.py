import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_solving_non_dcp_problem_raises_error(self) -> None:
    problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.Variable(pos=True) * cvxpy.Variable(pos=True)))
    with pytest.raises(error.DCPError, match='However, the problem does follow DGP rules'):
        problem.solve(SOLVER, gp=True)
        problem.solve(SOLVER)
    problem.solve(SOLVER, gp=True)
    self.assertEqual(problem.status, 'unbounded')
    self.assertAlmostEqual(problem.value, 0.0)