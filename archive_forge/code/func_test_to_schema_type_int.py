from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_to_schema_type_int(self):
    """Test Schema.to_schema_type method for type Integer."""
    schema = constraints.Schema('Integer')
    res = schema.to_schema_type(1)
    self.assertIsInstance(res, int)
    res = schema.to_schema_type('1')
    self.assertIsInstance(res, int)
    err = self.assertRaises(ValueError, schema.to_schema_type, 1.5)
    self.assertEqual('Value "1.5" is invalid for data type "Integer".', str(err))
    err = self.assertRaises(ValueError, schema.to_schema_type, '1.5')
    self.assertEqual('Value "1.5" is invalid for data type "Integer".', str(err))
    err = self.assertRaises(ValueError, schema.to_schema_type, 'foo')
    self.assertEqual('Value "foo" is invalid for data type "Integer".', str(err))