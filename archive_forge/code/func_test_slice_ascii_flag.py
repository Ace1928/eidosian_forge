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
def test_slice_ascii_flag(self):
    """
        Make sure ascii flag is False when ascii and non-ascii characters are
        mixed in output of Unicode slicing.
        """

    @njit
    def f(s):
        return (s[::2]._is_ascii, s[1::2]._is_ascii)
    s = '¿abc¡Y tú, quién te cre\t\tes?'
    self.assertEqual(f(s), (0, 1))