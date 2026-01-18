import re
import unittest
from oslo_config import types
def test_with_max_zero(self):
    t = types.Port(max=0)
    self.assertRaises(ValueError, t, 1)
    t(0)