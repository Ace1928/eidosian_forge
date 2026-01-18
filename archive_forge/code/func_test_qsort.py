import unittest
from ctypes import *
import _ctypes_test
def test_qsort(self):
    comparefunc = CFUNCTYPE(c_int, POINTER(c_char), POINTER(c_char))
    lib.my_qsort.argtypes = (c_void_p, c_size_t, c_size_t, comparefunc)
    lib.my_qsort.restype = None

    def sort(a, b):
        return three_way_cmp(a[0], b[0])
    chars = create_string_buffer(b'spam, spam, and spam')
    lib.my_qsort(chars, len(chars) - 1, sizeof(c_char), comparefunc(sort))
    self.assertEqual(chars.raw, b'   ,,aaaadmmmnpppsss\x00')