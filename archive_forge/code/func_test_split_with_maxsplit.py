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
def test_split_with_maxsplit(self):
    CASES = [(' a ', None, 1), ('', '⚡', 1), ('abcabc', '⚡', 1), ('🐍⚡', '⚡', 1), ('🐍⚡🐍', '⚡', 1), ('abababa', 'a', 2), ('abababa', 'b', 1), ('abababa', 'c', 2), ('abababa', 'ab', 1), ('abababa', 'aba', 5)]
    for pyfunc, fmt_str in [(split_with_maxsplit_usecase, "'%s'.split('%s', %d)?"), (split_with_maxsplit_kwarg_usecase, "'%s'.split('%s', maxsplit=%d)?")]:
        cfunc = njit(pyfunc)
        for test_str, splitter, maxsplit in CASES:
            self.assertEqual(pyfunc(test_str, splitter, maxsplit), cfunc(test_str, splitter, maxsplit), fmt_str % (test_str, splitter, maxsplit))