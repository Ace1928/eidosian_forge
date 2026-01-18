import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
def test_trivial_constraints_lb_conflict(self):
    """Test for violated trivial constraint lower bound."""
    with self.assertRaisesRegex(InfeasibleConstraintException, 'Trivial constraint c violates LB 2.0 ≤ BODY 1.'):
        self._trivial_constraints_lb_conflict()