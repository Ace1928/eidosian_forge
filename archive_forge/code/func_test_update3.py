import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
def test_update3(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    m.obj = pyo.Objective(expr=m.z)
    m.c1 = pyo.Constraint(expr=m.z >= m.x ** 2 + m.y ** 2)
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)
    opt.update()
    self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
    m.c2 = pyo.Constraint(expr=m.y >= m.x ** 2)
    opt.add_constraint(m.c2)
    self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
    opt.remove_constraint(m.c2)
    opt.update()
    self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)