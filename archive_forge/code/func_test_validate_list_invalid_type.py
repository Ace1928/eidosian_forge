import re
import unittest
from wsme import exc
from wsme import types
def test_validate_list_invalid_type(self):
    self.assertRaises(ValueError, types.validate_value, [int], 1)