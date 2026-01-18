import testtools
from ironicclient.v1 import resource_fields
def test_sort_excluded(self):
    foo = resource_fields.Resource(['item_3', 'item1', '2nd_item'], sort_excluded=['item1'])
    self.assertEqual(('item_3', '2nd_item'), foo.sort_fields)
    self.assertEqual(('Third item', 'A second item'), foo.sort_labels)