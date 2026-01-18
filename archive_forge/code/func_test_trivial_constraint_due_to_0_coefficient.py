import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
def test_trivial_constraint_due_to_0_coefficient(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.y.fix(0)
    m.c = Constraint(expr=m.x * m.y >= 0)
    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)
    self.assertFalse(m.c.active)