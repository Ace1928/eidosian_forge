import testtools
from ironicclient.v1 import resource_fields
def test_override_labels_unknown(self):
    self.assertRaises(ValueError, resource_fields.Resource, ['item_3', 'item1', '2nd_item'], override_labels={'foo': 'One'})