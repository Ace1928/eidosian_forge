import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
class TestDictColumn(utils.TestCase):

    def test_dict_column(self):
        data = {'key1': 'value1', 'key2': 'value2'}
        col = format_columns.DictColumn(data)
        self.assertEqual(data, col.machine_readable())
        self.assertEqual("key1='value1', key2='value2'", col.human_readable())

    def test_complex_object(self):
        """Non-dict objects should be converted to a dict."""
        data = collections.OrderedDict([('key1', 'value1'), ('key2', 'value2')])
        col = format_columns.DictColumn(data)
        self.assertEqual(type(col.machine_readable()), dict)