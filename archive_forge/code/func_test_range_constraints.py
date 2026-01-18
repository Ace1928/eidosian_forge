import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def test_range_constraints(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.xl = pe.Param(initialize=-1, mutable=True)
    m.xu = pe.Param(initialize=1, mutable=True)
    m.c = pe.Constraint(expr=pe.inequality(m.xl, m.x, m.xu))
    m.obj = pe.Objective(expr=m.x)
    opt = Gurobi()
    opt.set_instance(m)
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, -1)
    m.xl.value = -3
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, -3)
    del m.obj
    m.obj = pe.Objective(expr=m.x, sense=pe.maximize)
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, 1)
    m.xu.value = 3
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, 3)