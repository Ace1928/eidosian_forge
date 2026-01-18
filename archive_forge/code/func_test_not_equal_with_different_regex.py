import re
import unittest
from oslo_config import types
def test_not_equal_with_different_regex(self):
    t1 = types.String(regex=re.compile('^[A-Z]'))
    t2 = types.String(regex=re.compile('^[a-z]'))
    self.assertFalse(t1 == t2)