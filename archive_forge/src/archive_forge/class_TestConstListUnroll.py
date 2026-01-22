from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
class TestConstListUnroll(MemoryLeakMixin, TestCase):

    def test_01(self):

        @njit
        def foo():
            a = [12, 12.7, 3j, 4]
            acc = 0
            for i in range(len(literal_unroll(a))):
                acc += a[i]
                if acc.real < 26:
                    acc -= 1
                else:
                    break
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_02(self):

        @njit
        def foo():
            x = [12, 12.7, 3j, 4]
            acc = 0
            for a in literal_unroll(x):
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    break
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_03(self):

        @njit
        def foo():
            x = [12, 12.7, 3j, 4]
            y = ['foo', 8]
            acc = 0
            for a in literal_unroll(x):
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    for t in literal_unroll(y):
                        acc += t is False
                    break
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_04(self):

        @njit
        def foo():
            x = [12, 12.7, 3j, 4]
            y = ('foo', 8)
            acc = 0
            for a in literal_unroll(x):
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    for t in literal_unroll(y):
                        acc += t is False
                    break
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_05(self):

        @njit
        def foo(tup1, tup2):
            acc = 0
            for a in literal_unroll(tup1):
                if a[0] > 1:
                    acc += tup2[0].sum()
            return acc
        n = 10
        tup1 = [np.zeros(10), np.zeros(10)]
        tup2 = (np.ones((n,)), np.ones((n, n)), np.ones((n, n, n)), np.ones((n, n, n, n)), np.ones((n, n, n, n, n)))
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo(tup1, tup2)
        msg = 'Invalid use of literal_unroll with a function argument'
        self.assertIn(msg, str(raises.exception))

    def test_06(self):

        @njit
        def foo():
            n = 10
            tup = [np.ones((n,)), np.ones((n, n)), 'ABCDEFGHJI', (1, 2, 3), (1, 'foo', 2, 'bar'), {3, 4, 5, 6, 7}]
            acc = 0
            for a in literal_unroll(tup):
                acc += len(a)
            return acc
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo()
        self.assertIn('Found non-constant value at position 0', str(raises.exception))

    def test_7(self):

        def dt(value):
            if value == 'apple':
                return 1
            elif value == 'orange':
                return 2
            elif value == 'banana':
                return 3
            elif value == 3390155550:
                return 1554098974 + value

        @overload(dt, inline='always')
        def ol_dt(li):
            if isinstance(li, types.StringLiteral):
                value = li.literal_value
                if value == 'apple':

                    def impl(li):
                        return 1
                elif value == 'orange':

                    def impl(li):
                        return 2
                elif value == 'banana':

                    def impl(li):
                        return 3
                return impl
            elif isinstance(li, types.IntegerLiteral):
                value = li.literal_value
                if value == 3390155550:

                    def impl(li):
                        return 1554098974 + value
                    return impl

        @njit
        def foo():
            acc = 0
            for t in literal_unroll(['apple', 'orange', 'banana', 3390155550]):
                acc += dt(t)
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_8(self):

        @njit
        def foo():
            x = []
            z = ['apple', 'orange', 'banana']
            for i in range(len(literal_unroll(z))):
                t = z[i]
                if t == 'apple':
                    x.append('0')
                elif t == 'orange':
                    x.append(t)
                elif t == 'banana':
                    x.append('2.0')
            return x
        self.assertEqual(foo(), foo.py_func())

    def test_9(self):

        @njit
        def foo(idx, z):
            a = [12, 12.7, 3j, 4]
            acc = 0
            for i in literal_unroll(a):
                acc += i
                if acc.real < 26:
                    acc -= 1
                else:
                    for x in literal_unroll(a):
                        acc += x
                    break
            if a[0] < 23:
                acc += 2
            return acc
        f = 9
        k = f
        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_10(self):

        @njit
        def foo(idx, z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            b = [12, 12.7, 3j, 4]
            acc = 0
            for i in literal_unroll(a):
                acc += i
                if acc.real < 26:
                    acc -= 1
                else:
                    for x in literal_unroll(a):
                        for j in literal_unroll(b):
                            acc += j
                        acc += x
                for x in literal_unroll(a):
                    acc += x
            for x in literal_unroll(a):
                acc += x
            if a[0] < 23:
                acc += 2
            return acc
        f = 9
        k = f
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo(2, k)
        self.assertIn('Nesting of literal_unroll is unsupported', str(raises.exception))

    def test_11(self):

        @njit
        def foo():
            x = [1, 2, 3, 4]
            acc = 0
            for a in literal_unroll(x):
                acc += a
            return a
        self.assertEqual(foo(), foo.py_func())

    def test_12(self):

        @njit
        def foo():
            acc = 0
            x = [1, 2, 'a']
            for a in literal_unroll(x):
                acc += bool(a)
            return a
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        self.assertIn('Cannot unify', str(raises.exception))

    def test_13(self):

        @njit
        def foo():
            x = [1000, 2000, 3000, 4000]
            acc = 0
            for a in literal_unroll(x[:2]):
                acc += a
            return acc
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo()
        self.assertIn('Invalid use of literal_unroll', str(raises.exception))

    def test_14(self):

        @njit
        def foo():
            x = [1000, 2000, 3000, 4000]
            acc = 0
            for a in literal_unroll(x):
                acc += a
            x.append(10)
            return acc
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        self.assertIn("Unknown attribute 'append' of type Tuple", str(raises.exception))