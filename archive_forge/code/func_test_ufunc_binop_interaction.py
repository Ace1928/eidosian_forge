from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
@pytest.mark.xfail(IS_PYPY, reason='Bug in pypy3.{9, 10}-v7.3.13, #24862')
def test_ufunc_binop_interaction(self):
    ops = {'add': (np.add, True, float), 'sub': (np.subtract, True, float), 'mul': (np.multiply, True, float), 'truediv': (np.true_divide, True, float), 'floordiv': (np.floor_divide, True, float), 'mod': (np.remainder, True, float), 'divmod': (np.divmod, False, float), 'pow': (np.power, True, int), 'lshift': (np.left_shift, True, int), 'rshift': (np.right_shift, True, int), 'and': (np.bitwise_and, True, int), 'xor': (np.bitwise_xor, True, int), 'or': (np.bitwise_or, True, int), 'matmul': (np.matmul, True, float)}

    class Coerced(Exception):
        pass

    def array_impl(self):
        raise Coerced

    def op_impl(self, other):
        return 'forward'

    def rop_impl(self, other):
        return 'reverse'

    def iop_impl(self, other):
        return 'in-place'

    def array_ufunc_impl(self, ufunc, method, *args, **kwargs):
        return ('__array_ufunc__', ufunc, method, args, kwargs)

    def make_obj(base, array_priority=False, array_ufunc=False, alleged_module='__main__'):
        class_namespace = {'__array__': array_impl}
        if array_priority is not False:
            class_namespace['__array_priority__'] = array_priority
        for op in ops:
            class_namespace['__{0}__'.format(op)] = op_impl
            class_namespace['__r{0}__'.format(op)] = rop_impl
            class_namespace['__i{0}__'.format(op)] = iop_impl
        if array_ufunc is not False:
            class_namespace['__array_ufunc__'] = array_ufunc
        eval_namespace = {'base': base, 'class_namespace': class_namespace, '__name__': alleged_module}
        MyType = eval("type('MyType', (base,), class_namespace)", eval_namespace)
        if issubclass(MyType, np.ndarray):
            return np.arange(3, 7).reshape(2, 2).view(MyType)
        else:
            return MyType()

    def check(obj, binop_override_expected, ufunc_override_expected, inplace_override_expected, check_scalar=True):
        for op, (ufunc, has_inplace, dtype) in ops.items():
            err_msg = 'op: %s, ufunc: %s, has_inplace: %s, dtype: %s' % (op, ufunc, has_inplace, dtype)
            check_objs = [np.arange(3, 7, dtype=dtype).reshape(2, 2)]
            if check_scalar:
                check_objs.append(check_objs[0][0])
            for arr in check_objs:
                arr_method = getattr(arr, '__{0}__'.format(op))

                def first_out_arg(result):
                    if op == 'divmod':
                        assert_(isinstance(result, tuple))
                        return result[0]
                    else:
                        return result
                if binop_override_expected:
                    assert_equal(arr_method(obj), NotImplemented, err_msg)
                elif ufunc_override_expected:
                    assert_equal(arr_method(obj)[0], '__array_ufunc__', err_msg)
                elif isinstance(obj, np.ndarray) and type(obj).__array_ufunc__ is np.ndarray.__array_ufunc__:
                    res = first_out_arg(arr_method(obj))
                    assert_(res.__class__ is obj.__class__, err_msg)
                else:
                    assert_raises((TypeError, Coerced), arr_method, obj, err_msg=err_msg)
                arr_rmethod = getattr(arr, '__r{0}__'.format(op))
                if ufunc_override_expected:
                    res = arr_rmethod(obj)
                    assert_equal(res[0], '__array_ufunc__', err_msg=err_msg)
                    assert_equal(res[1], ufunc, err_msg=err_msg)
                elif isinstance(obj, np.ndarray) and type(obj).__array_ufunc__ is np.ndarray.__array_ufunc__:
                    res = first_out_arg(arr_rmethod(obj))
                    assert_(res.__class__ is obj.__class__, err_msg)
                else:
                    assert_raises((TypeError, Coerced), arr_rmethod, obj, err_msg=err_msg)
                if has_inplace and isinstance(arr, np.ndarray):
                    arr_imethod = getattr(arr, '__i{0}__'.format(op))
                    if inplace_override_expected:
                        assert_equal(arr_method(obj), NotImplemented, err_msg=err_msg)
                    elif ufunc_override_expected:
                        res = arr_imethod(obj)
                        assert_equal(res[0], '__array_ufunc__', err_msg)
                        assert_equal(res[1], ufunc, err_msg)
                        assert_(type(res[-1]['out']) is tuple, err_msg)
                        assert_(res[-1]['out'][0] is arr, err_msg)
                    elif isinstance(obj, np.ndarray) and type(obj).__array_ufunc__ is np.ndarray.__array_ufunc__:
                        assert_(arr_imethod(obj) is arr, err_msg)
                    else:
                        assert_raises((TypeError, Coerced), arr_imethod, obj, err_msg=err_msg)
                op_fn = getattr(operator, op, None)
                if op_fn is None:
                    op_fn = getattr(operator, op + '_', None)
                if op_fn is None:
                    op_fn = getattr(builtins, op)
                assert_equal(op_fn(obj, arr), 'forward', err_msg)
                if not isinstance(obj, np.ndarray):
                    if binop_override_expected:
                        assert_equal(op_fn(arr, obj), 'reverse', err_msg)
                    elif ufunc_override_expected:
                        assert_equal(op_fn(arr, obj)[0], '__array_ufunc__', err_msg)
                if ufunc_override_expected:
                    assert_equal(ufunc(obj, arr)[0], '__array_ufunc__', err_msg)
    check(make_obj(object), False, False, False)
    check(make_obj(object, array_priority=-2 ** 30), False, False, False)
    check(make_obj(object, array_priority=1), True, False, True)
    check(make_obj(np.ndarray, array_priority=1), False, False, False, check_scalar=False)
    check(make_obj(object, array_priority=1, array_ufunc=array_ufunc_impl), False, True, False)
    check(make_obj(np.ndarray, array_priority=1, array_ufunc=array_ufunc_impl), False, True, False)
    check(make_obj(object, array_ufunc=None), True, False, False)
    check(make_obj(np.ndarray, array_ufunc=None), True, False, False, check_scalar=False)