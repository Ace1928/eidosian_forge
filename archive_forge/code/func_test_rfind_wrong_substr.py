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
def test_rfind_wrong_substr(self):
    cfunc = njit(rfind_usecase)
    for s in UNICODE_EXAMPLES:
        for sub_str in [None, 1, False]:
            with self.assertRaises(TypingError) as raises:
                cfunc(s, sub_str)
            msg = 'must be {}'.format(types.UnicodeType)
            self.assertIn(msg, str(raises.exception))