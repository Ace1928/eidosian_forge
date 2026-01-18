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
def test_upper(self):

    def pyfunc(x):
        return x.upper()
    cfunc = njit(pyfunc)
    for a in UNICODE_EXAMPLES:
        args = [a]
        self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))