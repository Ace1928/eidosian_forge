import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
class TestListColumn(utils.TestCase):

    def test_list_column(self):
        data = ['key1', 'key2']
        col = format_columns.ListColumn(data)
        self.assertEqual(data, col.machine_readable())
        self.assertEqual('key1, key2', col.human_readable())

    def test_complex_object(self):
        """Non-list objects should be converted to a list."""
        data = {'key1', 'key2'}
        col = format_columns.ListColumn(data)
        self.assertEqual(type(col.machine_readable()), list)