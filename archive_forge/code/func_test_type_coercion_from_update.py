import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_type_coercion_from_update(self):

    def impl():
        i = np.uint64(1)
        R = set()
        R.update({1, 2, 3})
        R.add(i)
        return R
    check = self.unordered_checker(impl)
    check()