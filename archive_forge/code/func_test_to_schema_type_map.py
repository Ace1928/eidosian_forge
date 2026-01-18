from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_to_schema_type_map(self):
    """Test Schema.to_schema_type method for type Map."""
    schema = constraints.Schema('Map')
    res = schema.to_schema_type({'a': 'aa', 'b': 'bb'})
    self.assertIsInstance(res, dict)
    self.assertEqual({'a': 'aa', 'b': 'bb'}, res)