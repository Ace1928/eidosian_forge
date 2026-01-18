import re
import unittest
from wsme import exc
from wsme import types
def test_validate_list_invalid_member(self):
    self.assertRaises(ValueError, types.validate_value, [int], ['not-a-number'])