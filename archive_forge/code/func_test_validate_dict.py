import re
import unittest
from wsme import exc
from wsme import types
def test_validate_dict(self):
    assert types.validate_value({int: str}, {1: '1', 5: '5'})
    self.assertRaises(ValueError, types.validate_value, {int: str}, [])
    assert types.validate_value({int: str}, {'1': '1', 5: '5'})
    self.assertRaises(ValueError, types.validate_value, {int: str}, {1: 1, 5: '5'})