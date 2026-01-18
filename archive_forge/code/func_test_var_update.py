import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
@unittest.skipIf(not SolverFactory('glpk').available(), 'GLPK solver is not available.')
def test_var_update(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var(bounds=(0, 1))
    m.c = Constraint(expr=m.x == m.y)
    m.o = Objective(expr=m.x)
    TransformationFactory('contrib.aggregate_vars').apply_to(m)
    SolverFactory('glpk').solve(m)
    z = m._var_aggregator_info.z
    self.assertEqual(z[1].value, 0)
    self.assertEqual(m.x.value, None)
    self.assertEqual(m.y.value, None)
    TransformationFactory('contrib.aggregate_vars').update_variables(m)
    self.assertEqual(z[1].value, 0)
    self.assertEqual(m.x.value, 0)
    self.assertEqual(m.y.value, 0)