import re
import unittest
from oslo_config import types
def test_string_non_str_spaces(self):
    t = types.String()
    e = Exception(' bar ')
    self.assertEqual(['" bar "'], t.format_defaults('', sample_default=e))