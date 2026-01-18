import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_reflect_same_set(self):
    """
        When the same set object is reflected twice, behaviour should
        be consistent.
        """
    pyfunc = reflect_dual
    cfunc = jit(nopython=True)(pyfunc)
    pyset = set([1, 2, 3])
    cset = pyset.copy()
    expected = pyfunc(pyset, pyset)
    got = cfunc(cset, cset)
    self.assertPreciseEqual(expected, got)
    self.assertPreciseEqual(pyset, cset)
    self.assertRefCountEqual(pyset, cset)