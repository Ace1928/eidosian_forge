import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
class TestTuplePassing(TestCase):

    def test_unituple(self):
        tuple_type = types.UniTuple(types.int32, 2)
        cf_first = njit((tuple_type,))(tuple_first)
        cf_second = njit((tuple_type,))(tuple_second)
        self.assertPreciseEqual(cf_first((4, 5)), 4)
        self.assertPreciseEqual(cf_second((4, 5)), 5)

    def test_hetero_tuple(self):
        tuple_type = types.Tuple((types.int64, types.float32))
        cf_first = njit((tuple_type,))(tuple_first)
        cf_second = njit((tuple_type,))(tuple_second)
        self.assertPreciseEqual(cf_first((2 ** 61, 1.5)), 2 ** 61)
        self.assertPreciseEqual(cf_second((2 ** 61, 1.5)), 1.5)

    def test_size_mismatch(self):
        tuple_type = types.UniTuple(types.int32, 2)
        cfunc = njit((tuple_type,))(tuple_first)
        entry_point = cfunc.overloads[cfunc.signatures[0]].entry_point
        with self.assertRaises(ValueError) as raises:
            entry_point((4, 5, 6))
        self.assertEqual(str(raises.exception), 'size mismatch for tuple, expected 2 element(s) but got 3')