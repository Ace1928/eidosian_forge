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
def test_slice3(self):
    pyfunc = getitem_usecase
    cfunc = njit(pyfunc)
    for s in UNICODE_EXAMPLES:
        for i in range(-len(s), len(s)):
            for j in range(-len(s), len(s)):
                for k in [-2, -1, 1, 2]:
                    sl = slice(i, j, k)
                    self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d:%d]?" % (s, i, j, k))