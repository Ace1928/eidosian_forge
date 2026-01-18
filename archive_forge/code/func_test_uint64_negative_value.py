import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_uint64_negative_value(self):
    props = {'key': -1}
    with self.assertRaises(OverflowError):
        self._dict_to_nvlist_to_dict(props)