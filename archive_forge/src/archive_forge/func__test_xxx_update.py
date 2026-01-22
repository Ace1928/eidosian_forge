import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def _test_xxx_update(self, pyfunc):
    check = self.unordered_checker(pyfunc)
    sizes = (1, 50, 500)
    for na, nb in itertools.product(sizes, sizes):
        a = self.sparse_array(na)
        b = self.sparse_array(nb)
        check(a, b)