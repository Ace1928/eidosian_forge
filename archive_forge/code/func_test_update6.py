import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
def test_update6(self):
    m = pyo.ConcreteModel()
    m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
    m.x = pyo.Var(m.a, within=pyo.Binary)
    m.y = pyo.Var(within=pyo.Binary)
    m.obj = pyo.Objective(expr=m.y)
    m.c1 = pyo.SOSConstraint(var=m.x, sos=1)
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)
    opt.update()
    self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
    m.c2 = pyo.SOSConstraint(var=m.x, sos=2)
    opt.add_sos_constraint(m.c2)
    self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
    opt.remove_sos_constraint(m.c2)
    opt.update()
    self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)