import re
import unittest
from wsme import exc
from wsme import types
def test_validate_list_empty(self):
    assert types.validate_value([int], []) == []