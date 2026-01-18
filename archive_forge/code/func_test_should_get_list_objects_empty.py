import testtools
from barbicanclient import formatter
def test_should_get_list_objects_empty(self):
    columns, data = EntityFormatter._list_objects([])
    self.assertEqual([], columns)
    self.assertEqual([], [e for e in data])