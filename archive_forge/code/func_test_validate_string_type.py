import re
import unittest
from wsme import exc
from wsme import types
def test_validate_string_type(self):
    v = types.StringType(min_length=1, max_length=10, pattern='^[a-zA-Z0-9]*$')
    v.validate('1')
    v.validate('12345')
    v.validate('1234567890')
    self.assertRaises(ValueError, v.validate, '')
    self.assertRaises(ValueError, v.validate, '12345678901')
    v.validate('a')
    v.validate('A')
    self.assertRaises(ValueError, v.validate, '_')