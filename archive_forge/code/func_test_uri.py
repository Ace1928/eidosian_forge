import re
import unittest
from oslo_config import types
def test_uri(self):
    self.assertConvertedValue('http://example.com', 'http://example.com')
    self.assertInvalid('invalid')
    self.assertInvalid('http://')