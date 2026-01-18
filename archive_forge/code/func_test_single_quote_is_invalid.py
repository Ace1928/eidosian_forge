import re
import unittest
from oslo_config import types
def test_single_quote_is_invalid(self):
    self.type_instance = types.String(quotes=True)
    self.assertInvalid('"')
    self.assertInvalid("'")