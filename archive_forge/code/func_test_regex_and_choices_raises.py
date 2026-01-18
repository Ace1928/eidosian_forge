import re
import unittest
from oslo_config import types
def test_regex_and_choices_raises(self):
    self.assertRaises(ValueError, types.String, regex=re.compile('^[A-Z]'), choices=['Foo', 'Bar', 'baz'])