import re
import unittest
from wsme import exc
from wsme import types
def test_register_invalid_array(self):
    self.assertRaises(ValueError, types.register_type, [])
    self.assertRaises(ValueError, types.register_type, [int, str])
    self.assertRaises(AttributeError, types.register_type, [1])