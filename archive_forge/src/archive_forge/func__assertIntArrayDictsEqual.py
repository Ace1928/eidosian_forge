import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def _assertIntArrayDictsEqual(self, dict1, dict2):
    self.assertEqual(len(dict1), len(dict1), 'resulting dictionary is of different size')
    for key in dict1.keys():
        val1 = dict1[key]
        val2 = dict2[key]
        self.assertEqual(len(val1), len(val2), 'array values of different sizes')
        for x, y in zip(val1, val2):
            self.assertEqual(int(x), int(y))