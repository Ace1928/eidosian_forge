import re
import unittest
from wsme import exc
from wsme import types
def test_validate_float(self):
    self.assertEqual(types.validate_value(float, 1), 1.0)
    self.assertEqual(types.validate_value(float, '1'), 1.0)
    self.assertEqual(types.validate_value(float, 1.1), 1.1)
    self.assertRaises(ValueError, types.validate_value, float, [])
    self.assertRaises(ValueError, types.validate_value, float, 'not-a-float')