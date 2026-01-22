import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
class TestArrayLike:

    def setup_method(self):

        class MyArray:

            def __init__(self, function=None):
                self.function = function

            def __array_function__(self, func, types, args, kwargs):
                assert func is getattr(np, func.__name__)
                try:
                    my_func = getattr(self, func.__name__)
                except AttributeError:
                    return NotImplemented
                return my_func(*args, **kwargs)
        self.MyArray = MyArray

        class MyNoArrayFunctionArray:

            def __init__(self, function=None):
                self.function = function
        self.MyNoArrayFunctionArray = MyNoArrayFunctionArray

    def add_method(self, name, arr_class, enable_value_error=False):

        def _definition(*args, **kwargs):
            assert 'like' not in kwargs
            if enable_value_error and 'value_error' in kwargs:
                raise ValueError
            return arr_class(getattr(arr_class, name))
        setattr(arr_class, name, _definition)

    def func_args(*args, **kwargs):
        return (args, kwargs)

    def test_array_like_not_implemented(self):
        self.add_method('array', self.MyArray)
        ref = self.MyArray.array()
        with assert_raises_regex(TypeError, 'no implementation found'):
            array_like = np.asarray(1, like=ref)
    _array_tests = [('array', *func_args((1,))), ('asarray', *func_args((1,))), ('asanyarray', *func_args((1,))), ('ascontiguousarray', *func_args((2, 3))), ('asfortranarray', *func_args((2, 3))), ('require', *func_args((np.arange(6).reshape(2, 3),), requirements=['A', 'F'])), ('empty', *func_args((1,))), ('full', *func_args((1,), 2)), ('ones', *func_args((1,))), ('zeros', *func_args((1,))), ('arange', *func_args(3)), ('frombuffer', *func_args(b'\x00' * 8, dtype=int)), ('fromiter', *func_args(range(3), dtype=int)), ('fromstring', *func_args('1,2', dtype=int, sep=',')), ('loadtxt', *func_args(lambda: StringIO('0 1\n2 3'))), ('genfromtxt', *func_args(lambda: StringIO('1,2.1'), dtype=[('int', 'i8'), ('float', 'f8')], delimiter=','))]

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('numpy_ref', [True, False])
    def test_array_like(self, function, args, kwargs, numpy_ref):
        self.add_method('array', self.MyArray)
        self.add_method(function, self.MyArray)
        np_func = getattr(np, function)
        my_func = getattr(self.MyArray, function)
        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = self.MyArray.array()
        like_args = tuple((a() if callable(a) else a for a in args))
        array_like = np_func(*like_args, **kwargs, like=ref)
        if numpy_ref is True:
            assert type(array_like) is np.ndarray
            np_args = tuple((a() if callable(a) else a for a in args))
            np_arr = np_func(*np_args, **kwargs)
            if function == 'empty':
                np_arr.fill(1)
                array_like.fill(1)
            assert_equal(array_like, np_arr)
        else:
            assert type(array_like) is self.MyArray
            assert array_like.function is my_func

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('ref', [1, [1], 'MyNoArrayFunctionArray'])
    def test_no_array_function_like(self, function, args, kwargs, ref):
        self.add_method('array', self.MyNoArrayFunctionArray)
        self.add_method(function, self.MyNoArrayFunctionArray)
        np_func = getattr(np, function)
        if ref == 'MyNoArrayFunctionArray':
            ref = self.MyNoArrayFunctionArray.array()
        like_args = tuple((a() if callable(a) else a for a in args))
        with assert_raises_regex(TypeError, 'The `like` argument must be an array-like that implements'):
            np_func(*like_args, **kwargs, like=ref)

    @pytest.mark.parametrize('numpy_ref', [True, False])
    def test_array_like_fromfile(self, numpy_ref):
        self.add_method('array', self.MyArray)
        self.add_method('fromfile', self.MyArray)
        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = self.MyArray.array()
        data = np.random.random(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'testfile')
            data.tofile(fname)
            array_like = np.fromfile(fname, like=ref)
            if numpy_ref is True:
                assert type(array_like) is np.ndarray
                np_res = np.fromfile(fname, like=ref)
                assert_equal(np_res, data)
                assert_equal(array_like, np_res)
            else:
                assert type(array_like) is self.MyArray
                assert array_like.function is self.MyArray.fromfile

    def test_exception_handling(self):
        self.add_method('array', self.MyArray, enable_value_error=True)
        ref = self.MyArray.array()
        with assert_raises(TypeError):
            np.array(1, value_error=True, like=ref)

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    def test_like_as_none(self, function, args, kwargs):
        self.add_method('array', self.MyArray)
        self.add_method(function, self.MyArray)
        np_func = getattr(np, function)
        like_args = tuple((a() if callable(a) else a for a in args))
        like_args_exp = tuple((a() if callable(a) else a for a in args))
        array_like = np_func(*like_args, **kwargs, like=None)
        expected = np_func(*like_args_exp, **kwargs)
        if function == 'empty':
            array_like.fill(1)
            expected.fill(1)
        assert_equal(array_like, expected)