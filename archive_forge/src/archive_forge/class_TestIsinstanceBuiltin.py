import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
class TestIsinstanceBuiltin(TestCase):

    def test_isinstance(self):
        pyfunc = isinstance_usecase
        cfunc = jit(nopython=True)(pyfunc)
        inputs = (3, 5.0, 'Hello', b'world', 1j, [1, 2, 3], (1, 3, 3, 3), set([1, 2]), (1, 'nba', 2), None)
        for inpt in inputs:
            expected = pyfunc(inpt)
            got = cfunc(inpt)
            self.assertEqual(expected, got)

    def test_isinstance_dict(self):
        pyfunc = isinstance_dict
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(pyfunc(), cfunc())

    def test_isinstance_issue9125(self):
        pyfunc = invalid_isinstance_usecase_phi_nopropagate2
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(pyfunc(3), cfunc(3))

    def test_isinstance_numba_types(self):
        pyfunc = isinstance_usecase_numba_types
        cfunc = jit(nopython=True)(pyfunc)
        inputs = ((types.int32(1), 'int32'), (types.int64(2), 'int64'), (types.float32(3.0), 'float32'), (types.float64(4.0), 'float64'), (types.complex64(5j), 'no match'), (typed.List([1, 2]), 'typed list'), (typed.Dict.empty(types.int64, types.int64), 'typed dict'))
        for inpt, expected in inputs:
            got = cfunc(inpt)
            self.assertEqual(expected, got)

    def test_isinstance_numba_types_2(self):
        pyfunc = isinstance_usecase_numba_types_2
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(pyfunc(), cfunc())

    def test_isinstance_invalid_type(self):
        pyfunc = isinstance_usecase_invalid_type
        cfunc = jit(nopython=True)(pyfunc)
        self.assertTrue(cfunc(3.4))
        msg = 'Cannot infer numba type of python type'
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(100)
        self.assertIn(msg, str(raises.exception))

    def test_isinstance_exceptions(self):
        fns = [(invalid_isinstance_usecase, 'Cannot infer numba type of python type'), (invalid_isinstance_usecase_phi_nopropagate, 'isinstance() cannot determine the type of variable "z" due to a branch.'), (invalid_isinstance_optional_usecase, 'isinstance() cannot determine the type of variable "z" due to a branch.'), (invalid_isinstance_unsupported_type_usecase(), 'isinstance() does not support variables of type "ntpl(')]
        for fn, msg in fns:
            fn = njit(fn)
            with self.assertRaises(errors.TypingError) as raises:
                fn(100)
            self.assertIn(msg, str(raises.exception))

    def test_combinations(self):

        def gen_w_arg(clazz_type):

            def impl(x):
                return isinstance(x, clazz_type)
            return impl
        clazz_types = (int, float, complex, str, list, tuple, bytes, set, range, np.int8, np.float32)
        instances = (1, 2.3, 4j, '5', [6], (7,), b'8', {9}, None, (10, 11, 12), (13, 'a', 14j), np.array([15, 16, 17]), np.int8(18), np.float32(19), typed.Dict.empty(types.unicode_type, types.float64), typed.List.empty_list(types.complex128), np.ones(4))
        for ct in clazz_types:
            fn = njit(gen_w_arg(ct))
            for x in instances:
                expected = fn.py_func(x)
                got = fn(x)
                self.assertEqual(got, expected)

    def test_numba_types(self):

        def gen_w_arg(clazz_type):

            def impl():
                return isinstance(1, clazz_type)
            return impl
        clazz_types = (types.Integer, types.Float, types.Array)
        msg = 'Numba type classes.*are not supported'
        for ct in clazz_types:
            fn = njit(gen_w_arg(ct))
            with self.assertRaises(errors.TypingError) as raises:
                fn()
            self.assertRegex(str(raises.exception), msg)

    def test_python_numpy_scalar_alias_problem(self):

        @njit
        def foo():
            return isinstance(np.intp(10), int)
        self.assertEqual(foo(), True)
        self.assertEqual(foo.py_func(), False)

        @njit
        def bar():
            return isinstance(1, np.intp)
        self.assertEqual(bar(), True)
        self.assertEqual(bar.py_func(), False)

    def test_branch_prune(self):

        @njit
        def foo(x):
            if isinstance(x, str):
                return x + 'some_string'
            elif isinstance(x, complex):
                return np.imag(x)
            elif isinstance(x, tuple):
                return len(x)
            else:
                assert 0
        for x in ('string', 1 + 2j, ('a', 3, 4j)):
            expected = foo.py_func(x)
            got = foo(x)
            self.assertEqual(got, expected)