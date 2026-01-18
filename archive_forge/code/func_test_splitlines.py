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
def test_splitlines(self):
    pyfunc = splitlines_usecase
    cfunc = njit(pyfunc)
    cases = ['', '\n', 'abc\r\rabc\r\n', '🐍⚡\x0b', '\x0c🐍⚡\x0c\x0b\x0b🐍\x85', '\u2028aba\u2029baba', '\n\r\na\x0b\x0cb\x0b\x0cc\x1c\x1d\x1e']
    msg = 'Results of "{}".splitlines() must be equal'
    for s in cases:
        self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))