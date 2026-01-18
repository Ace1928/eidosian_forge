import re
import unittest
from oslo_config import types
def test_regex_matches_uncompiled(self):
    self.type_instance = types.String(regex='^[A-Z]')
    self.assertConvertedValue('Foo', 'Foo')