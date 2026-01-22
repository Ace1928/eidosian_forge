import re
import unittest
from oslo_config import types
def assertConvertedValue(self, s, expected):
    self.assertEqual(expected, self.type_instance(s))