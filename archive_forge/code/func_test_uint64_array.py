import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_uint64_array(self):
    props = {'key': [0, 1, 2 ** 64 - 1]}
    res = self._dict_to_nvlist_to_dict(props)
    self.assertEqual(props, res)