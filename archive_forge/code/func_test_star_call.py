from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
def test_star_call(self, objmode=False):
    """
        Test a function call with a *args.
        """
    cfunc, check = self.compile_func(star_call, objmode)
    check(1, (2,), (3,))