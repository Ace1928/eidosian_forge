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
def test_splitlines_with_keepends(self):
    pyfuncs = [splitlines_with_keepends_usecase, splitlines_with_keepends_kwarg_usecase]
    messages = ['Results of "{}".splitlines({}) must be equal', 'Results of "{}".splitlines(keepends={}) must be equal']
    cases = ['', '\n', 'abc\r\rabc\r\n', '🐍⚡\x0b', '\x0c🐍⚡\x0c\x0b\x0b🐍\x85', '\u2028aba\u2029baba', '\n\r\na\x0b\x0cb\x0b\x0cc\x1c\x1d\x1e']
    all_keepends = [True, False, 0, 1, -1, 100]
    for pyfunc, msg in zip(pyfuncs, messages):
        cfunc = njit(pyfunc)
        for s, keepends in product(cases, all_keepends):
            self.assertEqual(pyfunc(s, keepends), cfunc(s, keepends), msg=msg.format(s, keepends))