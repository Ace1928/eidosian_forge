import re
import unittest
from wsme import exc
from wsme import types
def test_validate_list_none(self):
    v = types.ArrayType(int)
    assert v.validate(None) is None