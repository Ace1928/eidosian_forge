from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_range_vals_invalid_type(self):
    self.assertRaises(exception.InvalidSchemaError, constraints.Range, '1', 10)
    self.assertRaises(exception.InvalidSchemaError, constraints.Range, 1, '10')