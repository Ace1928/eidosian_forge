import re
import unittest
from oslo_config import types
def test_regex_fails(self):
    self.type_instance = types.String(regex=re.compile('^[A-Z]'))
    self.assertInvalid('foo')