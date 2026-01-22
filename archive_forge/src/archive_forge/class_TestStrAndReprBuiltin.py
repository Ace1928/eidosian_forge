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
class TestStrAndReprBuiltin(MemoryLeakMixin, TestCase):

    def test_str_default(self):

        @njit
        def foo():
            return str()
        self.assertEqual(foo(), foo.py_func())

    def test_str_object_kwarg(self):

        @njit
        def foo(x):
            return str(object=x)
        value = 'a string'
        self.assertEqual(foo(value), foo.py_func(value))

    def test_str_calls_dunder_str(self):

        @njit
        def foo(x):
            return str(x)
        Dummy, DummyType = self.make_dummy_type()
        dummy = Dummy()
        string_repr = 'this is the dummy object str'
        Dummy.__str__ = lambda inst: string_repr

        @overload_method(DummyType, '__str__')
        def ol_dummy_string(dummy):

            def impl(dummy):
                return string_repr
            return impl

        @overload_method(DummyType, '__repr__')
        def ol_dummy_repr(dummy):

            def impl(dummy):
                return 'SHOULD NOT BE CALLED'
            return impl
        self.assertEqual(foo(dummy), foo.py_func(dummy))

    def test_str_falls_back_to_repr(self):

        @njit
        def foo(x):
            return str(x)
        Dummy, DummyType = self.make_dummy_type()
        dummy = Dummy()
        string_repr = 'this is the dummy object repr'
        Dummy.__repr__ = lambda inst: string_repr

        @overload_method(DummyType, '__repr__')
        def ol_dummy_repr(dummy):

            def impl(dummy):
                return string_repr
            return impl
        self.assertEqual(foo(dummy), foo.py_func(dummy))

    def test_repr(self):

        @njit
        def foo(x):
            return (repr(x), x)
        for x in ('abc', False, 123):
            self.assertEqual(foo(x), foo.py_func(x))

    def test_repr_fallback(self):
        Dummy, DummyType = self.make_dummy_type()
        dummy = Dummy()
        string_repr = f'<object type:{typeof(dummy)}>'
        Dummy.__repr__ = lambda inst: string_repr

        @box(DummyType)
        def box_dummy(typ, obj, c):
            clazobj = c.pyapi.unserialize(c.pyapi.serialize_object(Dummy))
            return c.pyapi.call_function_objargs(clazobj, ())

        @njit
        def foo(x):
            return str(x)
        self.assertEqual(foo(dummy), foo.py_func(dummy))