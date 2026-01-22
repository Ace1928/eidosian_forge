import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestSetsInvalid(TestSets):

    def symmetric_difference_usecase(a, b):
        s = a.symmetric_difference(b)
        return list(s)

    def difference_usecase(a, b):
        s = a.difference(b)
        return list(s)

    def intersection_usecase(a, b):
        s = a.intersection(b)
        return list(s)

    def union_usecase(a, b):
        s = a.union(b)
        return list(s)

    def _test_set_operator(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        a = set([1, 2, 4, 11])
        b = (1, 2, 3)
        msg = 'All arguments must be Sets'
        with self.assertRaisesRegex(TypingError, msg):
            cfunc(a, b)

    def test_difference(self):
        self._test_set_operator(TestSetsInvalid.difference_usecase)

    def test_intersection(self):
        self._test_set_operator(TestSetsInvalid.intersection_usecase)

    def test_symmetric_difference(self):
        self._test_set_operator(TestSetsInvalid.symmetric_difference_usecase)

    def test_union(self):
        self._test_set_operator(TestSetsInvalid.union_usecase)

    def make_operator_usecase(self, op):
        code = 'if 1:\n        def operator_usecase(a, b):\n            s = a %(op)s b\n            return list(s)\n        ' % dict(op=op)
        return compile_function('operator_usecase', code, globals())

    def make_inplace_operator_usecase(self, op):
        code = 'if 1:\n        def inplace_operator_usecase(a, b):\n            sa = a\n            sb = b\n            sc = sa\n            sc %(op)s sb\n            return list(sc), list(sa)\n        ' % dict(op=op)
        return compile_function('inplace_operator_usecase', code, globals())

    def make_comparison_usecase(self, op):
        code = 'if 1:\n        def comparison_usecase(a, b):\n            return set(a) %(op)s b\n        ' % dict(op=op)
        return compile_function('comparison_usecase', code, globals())