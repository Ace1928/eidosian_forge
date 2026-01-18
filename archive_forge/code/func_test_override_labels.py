import testtools
from ironicclient.v1 import resource_fields
def test_override_labels(self):
    foo = resource_fields.Resource(['item_3', 'item1', '2nd_item'], override_labels={'item1': 'One'})
    self.assertEqual(('Third item', 'One', 'A second item'), foo.labels)
    self.assertEqual(('Third item', 'One', 'A second item'), foo.sort_labels)