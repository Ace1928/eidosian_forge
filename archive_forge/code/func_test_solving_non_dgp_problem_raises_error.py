import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_solving_non_dgp_problem_raises_error(self) -> None:
    problem = cvxpy.Problem(cvxpy.Minimize(-1.0 * cvxpy.Variable()), [])
    with pytest.raises(error.DGPError, match='However, the problem does follow DCP rules'):
        problem.solve(SOLVER, gp=True)
    problem.solve(SOLVER)
    self.assertEqual(problem.status, 'unbounded')
    self.assertEqual(problem.value, -float('inf'))