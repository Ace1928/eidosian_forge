import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def test_nonconvex(self):
    if gurobipy.GRB.VERSION_MAJOR < 9:
        raise unittest.SkipTest
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.obj = pe.Objective(expr=m.x ** 2 + m.y ** 2)
    m.c = pe.Constraint(expr=m.y == (m.x - 1) ** 2 - 2)
    opt = Gurobi()
    opt.config.solver_options['nonconvex'] = 2
    opt.solve(m)
    self.assertAlmostEqual(m.x.value, -0.3660254037844423, 2)
    self.assertAlmostEqual(m.y.value, -0.13397459621555508, 2)