import re
import unittest
from oslo_config import types
def test_string_with_non_closed_quote_is_invalid(self):
    self.type_instance = types.String(quotes=True)
    self.assertInvalid('"foo bar')
    self.assertInvalid("'bar baz")