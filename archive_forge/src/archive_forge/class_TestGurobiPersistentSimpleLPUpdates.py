import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
class TestGurobiPersistentSimpleLPUpdates(unittest.TestCase):

    def setUp(self):
        self.m = pe.ConcreteModel()
        m = self.m
        m.x = pe.Var()
        m.y = pe.Var()
        m.p1 = pe.Param(mutable=True)
        m.p2 = pe.Param(mutable=True)
        m.p3 = pe.Param(mutable=True)
        m.p4 = pe.Param(mutable=True)
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c1 = pe.Constraint(expr=m.y - m.p1 * m.x >= m.p2)
        m.c2 = pe.Constraint(expr=m.y - m.p3 * m.x >= m.p4)

    def get_solution(self):
        try:
            import numpy as np
        except:
            raise unittest.SkipTest('numpy is not available')
        p1 = self.m.p1.value
        p2 = self.m.p2.value
        p3 = self.m.p3.value
        p4 = self.m.p4.value
        A = np.array([[1, -p1], [1, -p3]])
        rhs = np.array([p2, p4])
        sol = np.linalg.solve(A, rhs)
        x = float(sol[1])
        y = float(sol[0])
        return (x, y)

    def set_params(self, p1, p2, p3, p4):
        self.m.p1.value = p1
        self.m.p2.value = p2
        self.m.p3.value = p3
        self.m.p4.value = p4

    def test_lp(self):
        self.set_params(-1, -2, 0.1, -2)
        x, y = self.get_solution()
        opt = Gurobi()
        res = opt.solve(self.m)
        self.assertAlmostEqual(x + y, res.incumbent_objective)
        self.assertAlmostEqual(x + y, res.objective_bound)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertTrue(res.incumbent_objective is not None)
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)
        self.set_params(-1.25, -1, 0.5, -2)
        opt.config.load_solutions = False
        res = opt.solve(self.m)
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)
        x, y = self.get_solution()
        self.assertNotAlmostEqual(x, self.m.x.value)
        self.assertNotAlmostEqual(y, self.m.y.value)
        res.solution_loader.load_vars()
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)