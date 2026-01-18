import unittest
from numba import jit
from numba.core import types
def test_unsigned_int_return_type(self, flags=force_pyobj_flags):
    self.test_int_return_type(int_type=types.uint64, flags=flags)