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
def test_rpartition(self):
    pyfunc = rpartition_usecase
    cfunc = njit(pyfunc)
    CASES = [('', '⚡'), ('abcabc', '⚡'), ('🐍⚡', '⚡'), ('🐍⚡🐍', '⚡'), ('abababa', 'a'), ('abababa', 'b'), ('abababa', 'c'), ('abababa', 'ab'), ('abababa', 'aba')]
    msg = 'Results of "{}".rpartition("{}") must be equal'
    for s, sep in CASES:
        self.assertEqual(pyfunc(s, sep), cfunc(s, sep), msg=msg.format(s, sep))