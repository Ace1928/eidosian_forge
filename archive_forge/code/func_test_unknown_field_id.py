import testtools
from ironicclient.v1 import resource_fields
def test_unknown_field_id(self):
    self.assertRaises(KeyError, resource_fields.Resource, ['item1', 'unknown_id'])