import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def unordered_checker(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)

    def check(*args):
        expected = pyfunc(*args)
        got = cfunc(*args)
        self._assert_equal_unordered(expected, got)
    return check