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
def test_split_whitespace(self):
    pyfunc = split_whitespace_usecase
    cfunc = njit(pyfunc)
    all_whitespace = ''.join(map(chr, [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8232, 8233, 8239, 8287, 12288]))
    CASES = ['', 'abcabc', '🐍 ⚡', '🐍 ⚡ 🐍', '🐍   ⚡ 🐍  ', '  🐍   ⚡ 🐍', ' 🐍' + all_whitespace + '⚡ 🐍  ']
    for test_str in CASES:
        self.assertEqual(pyfunc(test_str), cfunc(test_str), "'%s'.split()?" % (test_str,))