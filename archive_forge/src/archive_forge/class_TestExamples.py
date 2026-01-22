import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestExamples(BaseTest):
    """
    Examples of using sets.
    """

    def test_unique(self):
        pyfunc = unique_usecase
        check = self.unordered_checker(pyfunc)
        check(self.duplicates_array(200))
        check(self.sparse_array(200))

    def test_type_coercion_from_update(self):

        def impl():
            i = np.uint64(1)
            R = set()
            R.update({1, 2, 3})
            R.add(i)
            return R
        check = self.unordered_checker(impl)
        check()