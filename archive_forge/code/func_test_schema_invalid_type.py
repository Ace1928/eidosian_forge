from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_schema_invalid_type(self):
    self.assertRaises(exception.InvalidSchemaError, constraints.Schema, 'String', schema=constraints.Schema('String'))