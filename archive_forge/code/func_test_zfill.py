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
def test_zfill(self):
    pyfunc = zfill_usecase
    cfunc = njit(pyfunc)
    ZFILL_INPUTS = ['ascii', '+ascii', '-ascii', '-asc ii-', '12345', '-12345', '+12345', '', '¡Y tú crs?', '🐍⚡', '+🐍⚡', '-🐍⚡', '大眼，小手。', '+大眼，小手。', '-大眼，小手。']
    with self.assertRaises(TypingError) as raises:
        cfunc(ZFILL_INPUTS[0], 1.1)
    self.assertIn('<width> must be an Integer', str(raises.exception))
    for s in ZFILL_INPUTS:
        for width in range(-3, 20):
            self.assertEqual(pyfunc(s, width), cfunc(s, width))