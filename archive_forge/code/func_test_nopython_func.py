import math
import unittest
from numba import jit
from numba.core import types
from numba.core.errors import TypingError, NumbaTypeError
def test_nopython_func(self, flags=enable_pyobj_flags):
    pyfunc = get_nopython_func
    cfunc = jit((), **flags)(pyfunc)
    if flags == enable_pyobj_flags:
        result = cfunc()
        self.assertEqual(result, abs)
    else:
        self.fail('Unexpected successful compilation.')