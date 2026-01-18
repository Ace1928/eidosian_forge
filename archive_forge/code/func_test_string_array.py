import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def test_string_array(self):
    props = {'key': ['value', 'value2']}
    res = self._dict_to_nvlist_to_dict(props)
    self.assertEqual(props, res)