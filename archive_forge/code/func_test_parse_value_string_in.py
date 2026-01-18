import json
import testtools
from troveclient.v1 import metadata
from unittest import mock
def test_parse_value_string_in(self):
    value = 'this is a string'
    new_value = self.metadata._parse_value(value)
    self.assertEqual(value, new_value)