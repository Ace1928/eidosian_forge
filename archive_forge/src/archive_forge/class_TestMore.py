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
class TestMore(TestCase):

    def test_invalid_use_of_unroller(self):

        @njit
        def foo():
            x = (10, 20)
            r = 0
            for a in literal_unroll(x, x):
                r += a
            return r
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo()
        self.assertIn('literal_unroll takes one argument, found 2', str(raises.exception))

    def test_non_constant_list(self):

        @njit
        def foo(y):
            x = [10, y]
            r = 0
            for a in literal_unroll(x):
                r += a
            return r
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo(10)
        self.assertIn('Found non-constant value at position 1 in a list argument to literal_unroll', str(raises.exception))

    @unittest.skip('numba.literally not supported yet')
    def test_literally_constant_list(self):
        from numba import literally

        @njit
        def foo(y):
            x = [10, literally(y)]
            r = 0
            for a in literal_unroll(x):
                r += a
            return r
        foo(12)

        @njit
        def bar():
            return foo(12)
        bar()

    @unittest.skip("inlining of foo doesn't have const prop so y isn't const")
    def test_inlined_unroll_list(self):

        @njit(inline='always')
        def foo(y):
            x = [10, y]
            r = 0
            for a in literal_unroll(x):
                r += a
            return r

        @njit
        def bar():
            return foo(12)
        self.assertEqual(bar(), 10 + 12)

    def test_unroll_tuple_arg(self):

        @njit
        def foo(y):
            x = (10, y)
            r = 0
            for a in literal_unroll(x):
                r += a
            return r
        self.assertEqual(foo(12), foo.py_func(12))
        self.assertEqual(foo(1.2), foo.py_func(1.2))

    def test_unroll_tuple_arg2(self):

        @njit
        def foo(x):
            r = 0
            for a in literal_unroll(x):
                r += a
            return r
        self.assertEqual(foo((12, 1.2)), foo.py_func((12, 1.2)))
        self.assertEqual(foo((12, 1.2)), foo.py_func((12, 1.2)))

    def test_unroll_tuple_alias(self):

        @njit
        def foo():
            x = (10, 1.2)
            out = 0
            for i in literal_unroll(x):
                j = i
                k = j
                out += j + k + i
            return out
        self.assertEqual(foo(), foo.py_func())

    def test_unroll_tuple_nested(self):

        @njit
        def foo():
            x = ((10, 1.2), (1j, 3.0))
            out = 0
            for i in literal_unroll(x):
                for j in i:
                    out += j
            return out
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        self.assertIn('getiter', str(raises.exception))
        re = '.*Tuple\\(int[0-9][0-9], float64\\).*'
        self.assertRegex(str(raises.exception), re)

    def test_unroll_tuple_of_dict(self):

        @njit
        def foo():
            x = {}
            x['a'] = 1
            x['b'] = 2
            y = {}
            y[3] = 'c'
            y[4] = 'd'
            for it in literal_unroll((x, y)):
                for k, v in it.items():
                    print(k, v)
        with captured_stdout() as stdout:
            foo()
        lines = stdout.getvalue().splitlines()
        self.assertEqual(lines, ['a 1', 'b 2', '3 c', '4 d'])

    def test_unroll_named_tuple(self):
        ABC = namedtuple('ABC', ['a', 'b', 'c'])

        @njit
        def foo():
            abc = ABC(1, 2j, 3.4)
            out = 0
            for i in literal_unroll(abc):
                out += i
            return out
        self.assertEqual(foo(), foo.py_func())

    def test_unroll_named_tuple_arg(self):
        ABC = namedtuple('ABC', ['a', 'b', 'c'])

        @njit
        def foo(x):
            out = 0
            for i in literal_unroll(x):
                out += i
            return out
        abc = ABC(1, 2j, 3.4)
        self.assertEqual(foo(abc), foo.py_func(abc))

    def test_unroll_named_unituple(self):
        ABC = namedtuple('ABC', ['a', 'b', 'c'])

        @njit
        def foo():
            abc = ABC(1, 2, 3)
            out = 0
            for i in literal_unroll(abc):
                out += i
            return out
        self.assertEqual(foo(), foo.py_func())

    def test_unroll_named_unituple_arg(self):
        ABC = namedtuple('ABC', ['a', 'b', 'c'])

        @njit
        def foo(x):
            out = 0
            for i in literal_unroll(x):
                out += i
            return out
        abc = ABC(1, 2, 3)
        self.assertEqual(foo(abc), foo.py_func(abc))

    def test_unroll_global_tuple(self):

        @njit
        def foo():
            out = 0
            for i in literal_unroll(_X_GLOBAL):
                out += i
            return out
        self.assertEqual(foo(), foo.py_func())

    def test_unroll_freevar_tuple(self):
        x = (10, 11)

        @njit
        def foo():
            out = 0
            for i in literal_unroll(x):
                out += i
            return out
        self.assertEqual(foo(), foo.py_func())

    def test_unroll_function_tuple(self):

        @njit
        def a():
            return 1

        @njit
        def b():
            return 2
        x = (a, b)

        @njit
        def foo():
            out = 0
            for f in literal_unroll(x):
                out += f()
            return out
        self.assertEqual(foo(), foo.py_func())

    def test_unroll_indexing_list(self):

        @njit
        def foo(cont):
            i = 0
            acc = 0
            normal_list = [a for a in cont]
            heter_tuple = ('a', 25, 0.23, None)
            for item in literal_unroll(heter_tuple):
                acc += normal_list[i]
                i += 1
                print(item)
            return (i, acc)
        data = [j for j in range(4)]
        with captured_stdout():
            self.assertEqual(foo(data), foo.py_func(data))
        with captured_stdout() as stdout:
            foo(data)
        lines = stdout.getvalue().splitlines()
        self.assertEqual(lines, ['a', '25', '0.23', 'None'])

    def test_unroller_as_freevar(self):
        mixed = (np.ones((1,)), np.ones((1, 1)), np.ones((1, 1, 1)))
        from numba import literal_unroll as freevar_unroll

        @njit
        def foo():
            out = 0
            for i in freevar_unroll(mixed):
                out += i.ndim
            return out
        self.assertEqual(foo(), foo.py_func())

    def test_unroll_with_non_conformant_loops_present(self):

        @njit('(Tuple((int64, float64)),)')
        def foo(tup):
            for t in literal_unroll(tup):
                pass
            x = 1
            while x == 1:
                x = 0

    def test_literal_unroll_legalize_var_names01(self):
        test = np.array([(1, 2), (2, 3)], dtype=[('a1', 'f8'), ('a2', 'f8')])
        fields = tuple(test.dtype.fields.keys())

        @njit
        def foo(arr):
            res = 0
            for k in literal_unroll(fields):
                res = res + np.abs(arr[k]).sum()
            return res
        self.assertEqual(foo(test), 8.0)

    def test_literal_unroll_legalize_var_names02(self):
        test = np.array([(1, 2), (2, 3)], dtype=[('a1[0]', 'f8'), ('a2[1]', 'f8')])
        fields = tuple(test.dtype.fields.keys())

        @njit
        def foo(arr):
            res = 0
            for k in literal_unroll(fields):
                res = res + np.abs(arr[k]).sum()
            return res
        self.assertEqual(foo(test), 8.0)