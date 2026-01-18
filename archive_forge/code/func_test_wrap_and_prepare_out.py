import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_wrap_and_prepare_out(self):

    class StoreArrayPrepareWrap(np.ndarray):
        _wrap_args = None
        _prepare_args = None

        def __new__(cls):
            return np.zeros(()).view(cls)

        def __array_wrap__(self, obj, context):
            self._wrap_args = context[1]
            return obj

        def __array_prepare__(self, obj, context):
            self._prepare_args = context[1]
            return obj

        @property
        def args(self):
            return (self._prepare_args, self._wrap_args)

        def __repr__(self):
            return 'a'

    def do_test(f_call, f_expected):
        a = StoreArrayPrepareWrap()
        f_call(a)
        p, w = a.args
        expected = f_expected(a)
        try:
            assert_equal(p, expected)
            assert_equal(w, expected)
        except AssertionError as e:
            raise AssertionError('\n'.join(['Bad arguments passed in ufunc call', ' expected:              {}'.format(expected), ' __array_prepare__ got: {}'.format(p), ' __array_wrap__ got:    {}'.format(w)]))
    do_test(lambda a: np.add(a, 0), lambda a: (a, 0))
    do_test(lambda a: np.add(a, 0, None), lambda a: (a, 0))
    do_test(lambda a: np.add(a, 0, out=None), lambda a: (a, 0))
    do_test(lambda a: np.add(a, 0, out=(None,)), lambda a: (a, 0))
    do_test(lambda a: np.add(0, 0, a), lambda a: (0, 0, a))
    do_test(lambda a: np.add(0, 0, out=a), lambda a: (0, 0, a))
    do_test(lambda a: np.add(0, 0, out=(a,)), lambda a: (0, 0, a))
    do_test(lambda a: np.add(a, 0, where=False), lambda a: (a, 0))
    do_test(lambda a: np.add(0, 0, a, where=False), lambda a: (0, 0, a))