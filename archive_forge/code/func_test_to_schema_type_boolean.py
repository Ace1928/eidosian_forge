from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_to_schema_type_boolean(self):
    """Test Schema.to_schema_type method for type Boolean."""
    schema = constraints.Schema('Boolean')
    true_values = [1, '1', True, 'true', 'True', 'yes', 'Yes']
    for v in true_values:
        res = schema.to_schema_type(v)
        self.assertIsInstance(res, bool)
        self.assertTrue(res)
    false_values = [0, '0', False, 'false', 'False', 'No', 'no']
    for v in false_values:
        res = schema.to_schema_type(v)
        self.assertIsInstance(res, bool)
        self.assertFalse(res)
    err = self.assertRaises(ValueError, schema.to_schema_type, 'foo')
    self.assertEqual('Value "foo" is invalid for data type "Boolean".', str(err))