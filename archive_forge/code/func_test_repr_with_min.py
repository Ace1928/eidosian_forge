import re
import unittest
from oslo_config import types
def test_repr_with_min(self):
    t = types.Port(min=123)
    self.assertEqual('Port(min=123, max=65535)', repr(t))