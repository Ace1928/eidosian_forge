import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
class TestTupleLengthError(unittest.TestCase):

    def test_tuple_length_error(self):

        @njit
        def eattuple(tup):
            return len(tup)
        with self.assertRaises(errors.UnsupportedError) as raises:
            tup = tuple(range(1001))
            eattuple(tup)
        expected = "Tuple 'tup' length must be smaller than 1000"
        self.assertIn(expected, str(raises.exception))