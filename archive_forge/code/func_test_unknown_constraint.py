from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_unknown_constraint(self):
    constraint = constraints.CustomConstraint('zero', environment=self.env)
    error = self.assertRaises(ValueError, constraint.validate, 1)
    self.assertEqual('"1" does not validate zero (constraint not found)', str(error))