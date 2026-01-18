import sys, unittest
from ctypes import *
def test_swapped(self):
    for typ in byteswapped_structures:
        self.assertEqual(typ.value.offset, 1)
        o = typ()
        o.value = 4
        self.assertEqual(o.value, 4)