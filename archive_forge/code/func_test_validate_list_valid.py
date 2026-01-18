import re
import unittest
from wsme import exc
from wsme import types
def test_validate_list_valid(self):
    assert types.validate_value([int], [1, 2])
    assert types.validate_value([int], ['5'])