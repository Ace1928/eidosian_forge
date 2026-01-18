import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def test_var_bounds(self):
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-1, 1))
    m.obj = pe.Objective(expr=m.x)
    opt = Gurobi()
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, -1)
    m.x.setlb(-3)
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, -3)
    del m.obj
    m.obj = pe.Objective(expr=m.x, sense=pe.maximize)
    opt = Gurobi()
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, 1)
    m.x.setub(3)
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, 3)