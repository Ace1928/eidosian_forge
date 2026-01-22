import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
def _trivial_constraints_ub_conflict(self):
    m = ConcreteModel()
    m.v1 = Var(initialize=1)
    m.c = Constraint(expr=m.v1 <= 0)
    m.v1.fix()
    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)