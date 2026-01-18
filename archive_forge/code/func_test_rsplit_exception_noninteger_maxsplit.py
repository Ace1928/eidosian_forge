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
def test_rsplit_exception_noninteger_maxsplit(self):
    pyfunc = rsplit_with_maxsplit_usecase
    cfunc = njit(pyfunc)
    accepted_types = (types.Integer, int)
    for sep in [' ', None]:
        with self.assertRaises(TypingError) as raises:
            cfunc('a', sep, 2.4)
        msg = '"maxsplit" must be {}, not float'.format(accepted_types)
        self.assertIn(msg, str(raises.exception))