import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest
def test_optional_to_optional(self):
    """
        Test error due mishandling of Optional to Optional casting

        Related issue: https://github.com/numba/numba/issues/1718
        """
    opt_int = types.Optional(types.intp)
    opt_flt = types.Optional(types.float64)
    sig = opt_flt(opt_int)

    @njit(sig)
    def foo(a):
        return a
    self.assertEqual(foo(2), 2)
    self.assertIsNone(foo(None))