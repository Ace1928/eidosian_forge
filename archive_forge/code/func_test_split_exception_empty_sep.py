from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
def test_split_exception_empty_sep(self):
    self.disable_leak_check()
    pyfunc = split_usecase
    cfunc = njit(pyfunc)
    for func in [pyfunc, cfunc]:
        with self.assertRaises(ValueError) as raises:
            func('a', '')
        self.assertIn('empty separator', str(raises.exception))