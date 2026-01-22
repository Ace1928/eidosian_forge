from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
class FormatSortParamTestCase(utils.TestCase):

    def test_format_sort_empty_input(self):
        for s in [None, '', []]:
            self.assertIsNone(cs.volumes._format_sort_param(s))

    def test_format_sort_string_single_key(self):
        s = 'id'
        self.assertEqual('id', cs.volumes._format_sort_param(s))

    def test_format_sort_string_single_key_and_dir(self):
        s = 'id:asc'
        self.assertEqual('id:asc', cs.volumes._format_sort_param(s))

    def test_format_sort_string_multiple(self):
        s = 'id:asc,status,size:desc'
        self.assertEqual('id:asc,status,size:desc', cs.volumes._format_sort_param(s))

    def test_format_sort_string_mappings(self):
        s = 'id:asc,name,size:desc'
        self.assertEqual('id:asc,display_name,size:desc', cs.volumes._format_sort_param(s))

    def test_format_sort_whitespace_trailing_comma(self):
        s = ' id : asc ,status,  size:desc,'
        self.assertEqual('id:asc,status,size:desc', cs.volumes._format_sort_param(s))

    def test_format_sort_list_of_strings(self):
        s = ['id:asc', 'status', 'size:desc']
        self.assertEqual('id:asc,status,size:desc', cs.volumes._format_sort_param(s))

    def test_format_sort_invalid_direction(self):
        for s in ['id:foo', 'id:asc,status,size:foo', ['id', 'status', 'size:foo']]:
            self.assertRaises(ValueError, cs.volumes._format_sort_param, s)