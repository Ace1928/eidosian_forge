from unittest import mock
import testscenarios
import testtools
import yaml
from heatclient.common import template_format
class DetailedYAMLParseExceptions(testtools.TestCase):

    def test_parse_to_value_exception(self):
        yaml = 'not important\nbut very:\n  - incorrect\n'
        ex = self.assertRaises(ValueError, template_format.parse, yaml)
        self.assertIn('but very:\n            ^', str(ex))