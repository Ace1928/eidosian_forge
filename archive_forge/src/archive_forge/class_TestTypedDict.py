import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
class TestTypedDict(MemoryLeakMixin, TestCase):

    def test_basic(self):
        d = Dict.empty(int32, float32)
        self.assertEqual(len(d), 0)
        d[1] = 1
        d[2] = 2.3
        d[3] = 3.4
        self.assertEqual(len(d), 3)
        self.assertEqual(list(d.keys()), [1, 2, 3])
        for x, y in zip(list(d.values()), [1, 2.3, 3.4]):
            self.assertAlmostEqual(x, y, places=4)
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[2], 2.3, places=4)
        self.assertAlmostEqual(d[3], 3.4, places=4)
        del d[2]
        self.assertEqual(len(d), 2)
        self.assertIsNone(d.get(2))
        d.setdefault(2, 100)
        d.setdefault(3, 200)
        self.assertEqual(d[2], 100)
        self.assertAlmostEqual(d[3], 3.4, places=4)
        d.update({4: 5, 5: 6})
        self.assertAlmostEqual(d[4], 5)
        self.assertAlmostEqual(d[5], 6)
        self.assertTrue(4 in d)
        pyd = dict(d.items())
        self.assertEqual(len(pyd), len(d))
        self.assertAlmostEqual(d.pop(4), 5)
        nelem = len(d)
        k, v = d.popitem()
        self.assertEqual(len(d), nelem - 1)
        self.assertTrue(k not in d)
        copied = d.copy()
        self.assertEqual(copied, d)
        self.assertEqual(list(copied.items()), list(d.items()))

    def test_copy_from_dict(self):
        expect = {k: float(v) for k, v in zip(range(10), range(10, 20))}
        nbd = Dict.empty(int32, float64)
        for k, v in expect.items():
            nbd[k] = v
        got = dict(nbd)
        self.assertEqual(got, expect)

    def test_compiled(self):

        @njit
        def producer():
            d = Dict.empty(int32, float64)
            d[1] = 1.23
            return d

        @njit
        def consumer(d):
            return d[1]
        d = producer()
        val = consumer(d)
        self.assertEqual(val, 1.23)

    def test_gh7908(self):
        d = Dict.empty(key_type=types.Tuple([types.uint32, types.uint32]), value_type=int64)
        d[1, 1] = 12345
        self.assertEqual(d[1, 1], d.get((1, 1)))

    def check_stringify(self, strfn, prefix=False):
        nbd = Dict.empty(int32, int32)
        d = {}
        nbd[1] = 2
        d[1] = 2
        checker = self.assertIn if prefix else self.assertEqual
        checker(strfn(d), strfn(nbd))
        nbd[2] = 3
        d[2] = 3
        checker(strfn(d), strfn(nbd))
        for i in range(10, 20):
            nbd[i] = i + 1
            d[i] = i + 1
        checker(strfn(d), strfn(nbd))
        if prefix:
            self.assertTrue(strfn(nbd).startswith('DictType'))

    def test_repr(self):
        self.check_stringify(repr, prefix=True)

    def test_str(self):
        self.check_stringify(str)