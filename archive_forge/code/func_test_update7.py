import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
def test_update7(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)
    self.assertEqual(opt._solver_model.getAttr('NumVars'), 0)
    opt.remove_var(m.x)
    opt.update()
    self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)
    opt.add_var(m.x)
    self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)
    opt.update()
    self.assertEqual(opt._solver_model.getAttr('NumVars'), 2)
    opt.remove_var(m.x)
    opt.update()
    opt.add_var(m.x)
    opt.remove_var(m.x)
    opt.update()
    self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)