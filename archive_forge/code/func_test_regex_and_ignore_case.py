import re
import unittest
from oslo_config import types
def test_regex_and_ignore_case(self):
    self.type_instance = types.String(regex=re.compile('^[A-Z]'), ignore_case=True)
    self.assertConvertedValue('foo', 'foo')