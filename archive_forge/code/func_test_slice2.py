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
def test_slice2(self):
    pyfunc = getitem_usecase
    cfunc = njit(pyfunc)
    for s in UNICODE_EXAMPLES:
        for i in list(range(-len(s), len(s))):
            for j in list(range(-len(s), len(s))):
                sl = slice(i, j)
                self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d]?" % (s, i, j))