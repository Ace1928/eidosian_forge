import re
import unittest
from oslo_config import types
def test_repr_with_min_and_max(self):
    t = types.Port(min=123, max=456)
    self.assertEqual('Port(min=123, max=456)', repr(t))
    t = types.Port(min=0, max=0)
    self.assertEqual('Port(min=0, max=0)', repr(t))