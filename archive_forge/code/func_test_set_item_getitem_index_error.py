import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_set_item_getitem_index_error(self):
    l = List(self, 8, 0)
    with self.assertRaises(IndexError):
        l[0]
    with self.assertRaises(IndexError):
        l[0] = b'abcdefgh'