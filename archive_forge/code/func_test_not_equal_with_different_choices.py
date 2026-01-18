import re
import unittest
from oslo_config import types
def test_not_equal_with_different_choices(self):
    t1 = types.String(choices=['foo', 'bar'])
    t2 = types.String(choices=['foo', 'baz'])
    t3 = types.String(choices=('foo', 'baz'))
    self.assertFalse(t1 == t2)
    self.assertFalse(t1 == t3)