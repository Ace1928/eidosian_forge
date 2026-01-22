import unittest
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
from cvxpy.tests.base_test import BaseTest
@unittest.skipUnless(len(MIP_SOLVERS) > 0, 'No mixed-integer solver is installed.')
class TestMIPVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """

    def setUp(self) -> None:
        self.x_bool = cp.Variable(boolean=True)
        self.y_int = cp.Variable(integer=True)
        self.A_bool = cp.Variable((3, 2), boolean=True)
        self.B_int = cp.Variable((2, 3), integer=True)
        self.solvers = MIP_SOLVERS

    def test_all_solvers(self) -> None:
        for solver in self.solvers:
            self.bool_prob(solver)
            if solver != cp.SCIPY:
                self.int_prob(solver)
            if solver in [cp.CPLEX, cp.GUROBI, cp.MOSEK, cp.XPRESS]:
                if solver != cp.XPRESS:
                    self.bool_socp(solver)
                self.int_socp(solver)

    def bool_prob(self, solver) -> None:
        obj = cp.Minimize(cp.abs(self.x_bool - 0.2))
        p = cp.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.2)
        self.assertAlmostEqual(self.x_bool.value, 0)
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [cp.abs(self.x_bool) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)
        self.assertAlmostEqual(self.x_bool.value, 0, places=4)
        C = np.array([[0, 1, 0], [1, 1, 1]]).T
        obj = cp.Minimize(cp.sum(cp.abs(self.A_bool - C)))
        p = cp.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)
        self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [cp.sum(cp.abs(self.A_bool - C)) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)
        self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)

    def int_prob(self, solver) -> None:
        obj = cp.Minimize(cp.abs(self.y_int - 0.2))
        p = cp.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.2)
        self.assertAlmostEqual(self.y_int.value, 0)
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [self.y_int == 0.5, t >= 0])
        result = p.solve(solver=solver)
        self.assertEqual(p.status in s.INF_OR_UNB, True)

    def int_socp(self, solver) -> None:
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [cp.square(self.y_int - 0.2) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.04)
        self.assertAlmostEqual(self.y_int.value, 0)

    def bool_socp(self, solver) -> None:
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [cp.square(self.x_bool - 0.2) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.04)
        self.assertAlmostEqual(self.x_bool.value, 0)