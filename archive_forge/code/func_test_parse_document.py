from unittest import mock
import yaml
from heat.common import environment_format
from heat.tests import common
def test_parse_document(self):
    env = '["foo" , "bar"]'
    expect = 'The environment is not a valid YAML mapping data type.'
    msg = self.assertRaises(ValueError, environment_format.parse, env)
    self.assertIn(expect, msg.args)