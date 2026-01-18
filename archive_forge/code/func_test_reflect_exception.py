import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_reflect_exception(self):
    """
        When the function exits with an exception, sets should still be
        reflected.
        """
    pyfunc = reflect_exception
    cfunc = jit(nopython=True)(pyfunc)
    s = set([1, 2, 3])
    with self.assertRefCount(s):
        with self.assertRaises(ZeroDivisionError):
            cfunc(s)
        self.assertPreciseEqual(s, set([1, 2, 3, 42]))