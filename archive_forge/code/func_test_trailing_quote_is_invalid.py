import re
import unittest
from oslo_config import types
def test_trailing_quote_is_invalid(self):
    self.assertInvalid('foo.bar"')