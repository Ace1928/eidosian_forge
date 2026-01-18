import re
import unittest
from oslo_config import types
def test_strip_trailing_dot(self):
    self.assertConvertedValue('cell1.nova.site1.', 'cell1.nova.site1')
    self.assertConvertedValue('cell1.', 'cell1')