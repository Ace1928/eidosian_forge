import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
def test_var_attr(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.Binary)
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)
    opt.set_var_attr(m.x, 'Start', 1)
    self.assertEqual(opt.get_var_attr(m.x, 'Start'), 1)