import re
import unittest
from oslo_config import types
def test_range_bounds(self):
    self.type_instance = types.Range(1, 3)
    self.assertRange('1-3', 1, 4)
    self.assertRange('2-2', 2, 3)
    self.assertRange('2', 2, 3)
    self.assertInvalid('1-4')
    self.assertInvalid('0-3')
    self.assertInvalid('0-4')