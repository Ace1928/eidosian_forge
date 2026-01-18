import re
import unittest
from oslo_config import types
def test_range_exclusive(self):
    self.type_instance = types.Range(inclusive=False)
    self.assertRange('0-2', 0, 2)
    self.assertRange('-2-0', -2, 0)
    self.assertRange('2-0', 2, 0, -1)
    self.assertRange('-3--1', -3, -1)
    self.assertRange('-1--3', -1, -3, -1)
    self.assertRange('-1', -1, -1)