import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestSetsInvalidDtype(TestSets):

    def _test_set_operator(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        a = set([1, 2, 4, 11])
        b = set(['a', 'b', 'c'])
        msg = 'All Sets must be of the same type'
        with self.assertRaisesRegex(TypingError, msg):
            cfunc(a, b)