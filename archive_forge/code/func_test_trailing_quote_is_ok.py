import re
import unittest
from oslo_config import types
def test_trailing_quote_is_ok(self):
    self.type_instance = types.String(quotes=True)
    self.assertConvertedValue('foo bar"', 'foo bar"')