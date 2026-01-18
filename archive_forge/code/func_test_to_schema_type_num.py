from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_to_schema_type_num(self):
    """Test Schema.to_schema_type method for type Number."""
    schema = constraints.Schema('Number')
    res = schema.to_schema_type(1)
    self.assertIsInstance(res, int)
    res = schema.to_schema_type('1')
    self.assertIsInstance(res, int)
    res = schema.to_schema_type(1.5)
    self.assertIsInstance(res, float)
    res = schema.to_schema_type('1.5')
    self.assertIsInstance(res, float)
    self.assertEqual(1.5, res)
    err = self.assertRaises(ValueError, schema.to_schema_type, 'foo')
    self.assertEqual('Value "foo" is invalid for data type "Number".', str(err))