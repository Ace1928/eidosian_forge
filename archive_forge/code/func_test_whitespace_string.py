import re
import unittest
from oslo_config import types
def test_whitespace_string(self):
    self.assertConvertedValue('   \t\t\t\t', None)