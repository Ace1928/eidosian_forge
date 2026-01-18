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
def test_rfind_wrong_start_end_optional(self):
    s = UNICODE_EXAMPLES[0]
    sub_str = s[1:-1]
    accepted_types = (types.Integer, types.NoneType)
    msg = 'must be {}'.format(accepted_types)

    def try_compile_wrong_start_optional(*args):
        wrong_sig_optional = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.float64), types.Optional(types.intp))
        njit([wrong_sig_optional])(rfind_with_start_end_usecase)
    with self.assertRaises(TypingError) as raises:
        try_compile_wrong_start_optional(s, sub_str, 0.1, 1)
    self.assertIn(msg, str(raises.exception))

    def try_compile_wrong_end_optional(*args):
        wrong_sig_optional = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.intp), types.Optional(types.float64))
        njit([wrong_sig_optional])(rfind_with_start_end_usecase)
    with self.assertRaises(TypingError) as raises:
        try_compile_wrong_end_optional(s, sub_str, 1, 0.1)
    self.assertIn(msg, str(raises.exception))