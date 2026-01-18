import json
import testtools
from troveclient.v1 import metadata
from unittest import mock
def test_parse_value_valid_json_in(self):
    value = {'one': [2, 3, 4]}
    ser_value = json.dumps(value)
    new_value = self.metadata._parse_value(ser_value)
    self.assertEqual(value, new_value)