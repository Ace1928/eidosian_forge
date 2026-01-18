import re
import unittest
from wsme import exc
from wsme import types
def test_validate_string_type_precompile(self):
    precompile = re.compile('^[a-zA-Z0-9]*$')
    v = types.StringType(min_length=1, max_length=10, pattern=precompile)
    v.validate('a')
    v.validate('A')
    self.assertRaises(ValueError, v.validate, '_')