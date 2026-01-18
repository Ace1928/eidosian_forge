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