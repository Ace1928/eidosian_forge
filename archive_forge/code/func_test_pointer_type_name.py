import unittest, sys
from ctypes import *
import _ctypes_test
def test_pointer_type_name(self):
    LargeNamedType = type('T' * 2 ** 25, (Structure,), {})
    self.assertTrue(POINTER(LargeNamedType))
    from ctypes import _pointer_type_cache
    del _pointer_type_cache[LargeNamedType]