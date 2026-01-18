from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
@unittest.skipUnless(gurobi_available, 'Gurobi is not available')
def test_reverse_implication_for_land(self):
    m = ConcreteModel()
    m.t = BooleanVar()
    m.a = BooleanVar()
    m.d = BooleanVar()
    m.c = LogicalConstraint(expr=m.t.equivalent_to(m.a.land(m.d)))
    m.a.fix(True)
    m.d.fix(True)
    m.binary = Var(domain=Binary)
    m.t.associate_binary_var(m.binary)
    m.obj = Objective(expr=m.binary)
    TransformationFactory('contrib.logical_to_disjunctive').apply_to(m)
    TransformationFactory('gdp.bigm').apply_to(m)
    SolverFactory('gurobi').solve(m)
    update_boolean_vars_from_binary(m)
    self.assertEqual(value(m.obj), 1)
    self.assertTrue(value(m.t))