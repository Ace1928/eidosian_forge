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
def _check_ordering_op(self, usecase):
    pyfunc = usecase
    cfunc = njit(pyfunc)
    for a in UNICODE_ORDERING_EXAMPLES:
        self.assertEqual(pyfunc(a, a), cfunc(a, a), '%s: "%s", "%s"' % (usecase.__name__, a, a))
    for a, b in permutations(UNICODE_ORDERING_EXAMPLES, r=2):
        self.assertEqual(pyfunc(a, b), cfunc(a, b), '%s: "%s", "%s"' % (usecase.__name__, a, b))
        self.assertEqual(pyfunc(b, a), cfunc(b, a), '%s: "%s", "%s"' % (usecase.__name__, b, a))