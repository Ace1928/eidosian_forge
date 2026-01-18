import re
import unittest
from oslo_config import types
def test_repr_with_max(self):
    t = types.Port(max=456)
    self.assertEqual('Port(min=0, max=456)', repr(t))