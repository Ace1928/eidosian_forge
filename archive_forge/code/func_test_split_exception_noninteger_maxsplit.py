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
def test_split_exception_noninteger_maxsplit(self):
    pyfunc = split_with_maxsplit_usecase
    cfunc = njit(pyfunc)
    for sep in [' ', None]:
        with self.assertRaises(TypingError) as raises:
            cfunc('a', sep, 2.4)
        self.assertIn('float64', str(raises.exception), 'non-integer maxsplit with sep = %s' % sep)