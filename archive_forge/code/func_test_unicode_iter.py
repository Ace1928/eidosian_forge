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
def test_unicode_iter(self):
    pyfunc = iter_usecase
    cfunc = njit(pyfunc)
    for a in UNICODE_EXAMPLES:
        self.assertPreciseEqual(pyfunc(a), cfunc(a))