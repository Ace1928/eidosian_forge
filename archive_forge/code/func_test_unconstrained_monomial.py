import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_unconstrained_monomial(self) -> None:
    x = cvxpy.Variable(pos=True)
    y = cvxpy.Variable(pos=True)
    prod = x * y
    dgp = cvxpy.Problem(cvxpy.Minimize(prod), [])
    dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
    dcp = dgp2dcp.reduce()
    self.assertIsInstance(dcp.objective.expr, AddExpression)
    self.assertEqual(len(dcp.objective.expr.args), 2)
    self.assertIsInstance(dcp.objective.expr.args[0], cvxpy.Variable)
    self.assertIsInstance(dcp.objective.expr.args[1], cvxpy.Variable)
    opt = dcp.solve(SOLVER)
    self.assertEqual(opt, -float('inf'))
    self.assertEqual(dcp.status, 'unbounded')
    dgp.unpack(dgp2dcp.retrieve(dcp.solution))
    self.assertAlmostEqual(dgp.value, 0.0)
    self.assertEqual(dgp.status, 'unbounded')
    dgp._clear_solution()
    dgp.solve(SOLVER, gp=True)
    self.assertAlmostEqual(dgp.value, 0.0)
    self.assertEqual(dgp.status, 'unbounded')
    dgp = cvxpy.Problem(cvxpy.Maximize(prod), [])
    dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
    dcp = dgp2dcp.reduce()
    self.assertEqual(dcp.solve(SOLVER), float('inf'))
    self.assertEqual(dcp.status, 'unbounded')
    dgp.unpack(dgp2dcp.retrieve(dcp.solution))
    self.assertEqual(dgp.value, float('inf'))
    self.assertEqual(dgp.status, 'unbounded')
    dgp._clear_solution()
    dgp.solve(SOLVER, gp=True)
    self.assertAlmostEqual(dgp.value, float('inf'))
    self.assertEqual(dgp.status, 'unbounded')