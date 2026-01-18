import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_uint64_array_too_large_value(self):
    props = {'key': [0, 2 ** 64]}
    with self.assertRaises(OverflowError):
        self._dict_to_nvlist_to_dict(props)