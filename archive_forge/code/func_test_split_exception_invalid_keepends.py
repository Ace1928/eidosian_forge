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
def test_split_exception_invalid_keepends(self):
    pyfunc = splitlines_with_keepends_usecase
    cfunc = njit(pyfunc)
    accepted_types = (types.Integer, int, types.Boolean, bool)
    for ty, keepends in (('none', None), ('unicode_type', 'None')):
        with self.assertRaises(TypingError) as raises:
            cfunc('\n', keepends)
        msg = '"keepends" must be {}, not {}'.format(accepted_types, ty)
        self.assertIn(msg, str(raises.exception))