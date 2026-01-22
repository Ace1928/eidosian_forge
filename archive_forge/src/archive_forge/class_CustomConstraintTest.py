from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
class CustomConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(CustomConstraintTest, self).setUp()
        self.env = environment.Environment({})

    def test_validation(self):

        class ZeroConstraint(object):

            def validate(self, value, context):
                return value == 0
        self.env.register_constraint('zero', ZeroConstraint)
        constraint = constraints.CustomConstraint('zero', environment=self.env)
        self.assertEqual('Value must be of type zero', str(constraint))
        self.assertIsNone(constraint.validate(0))
        error = self.assertRaises(ValueError, constraint.validate, 1)
        self.assertEqual('"1" does not validate zero', str(error))

    def test_custom_error(self):

        class ZeroConstraint(object):

            def error(self, value):
                return '%s is not 0' % value

            def validate(self, value, context):
                return value == 0
        self.env.register_constraint('zero', ZeroConstraint)
        constraint = constraints.CustomConstraint('zero', environment=self.env)
        error = self.assertRaises(ValueError, constraint.validate, 1)
        self.assertEqual('1 is not 0', str(error))

    def test_custom_message(self):

        class ZeroConstraint(object):
            message = 'Only zero!'

            def validate(self, value, context):
                return value == 0
        self.env.register_constraint('zero', ZeroConstraint)
        constraint = constraints.CustomConstraint('zero', environment=self.env)
        self.assertEqual('Only zero!', str(constraint))

    def test_unknown_constraint(self):
        constraint = constraints.CustomConstraint('zero', environment=self.env)
        error = self.assertRaises(ValueError, constraint.validate, 1)
        self.assertEqual('"1" does not validate zero (constraint not found)', str(error))

    def test_constraints(self):

        class ZeroConstraint(object):

            def validate(self, value, context):
                return value == 0
        self.env.register_constraint('zero', ZeroConstraint)
        constraint = constraints.CustomConstraint('zero', environment=self.env)
        self.assertEqual('zero', constraint['custom_constraint'])