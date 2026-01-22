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
class TestArrayFunctionImplementation:

    def test_one_arg(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(dispatched_one_arg)
        def _(array):
            return 'myarray'
        assert_equal(dispatched_one_arg(1), 'original')
        assert_equal(dispatched_one_arg(MyArray()), 'myarray')

    def test_optional_args(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array, option=None: (array,))
        def func_with_option(array, option='default'):
            return option

        @implements(func_with_option)
        def my_array_func_with_option(array, new_option='myarray'):
            return new_option
        assert_equal(func_with_option(1), 'default')
        assert_equal(func_with_option(1, option='extra'), 'extra')
        assert_equal(func_with_option(MyArray()), 'myarray')
        with assert_raises(TypeError):
            func_with_option(MyArray(), option='extra')
        result = my_array_func_with_option(MyArray(), new_option='yes')
        assert_equal(result, 'yes')
        with assert_raises(TypeError):
            func_with_option(MyArray(), new_option='no')

    def test_not_implemented(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array: (array,), module='my')
        def func(array):
            return array
        array = np.array(1)
        assert_(func(array) is array)
        assert_equal(func.__module__, 'my')
        with assert_raises_regex(TypeError, "no implementation found for 'my.func'"):
            func(MyArray())

    @pytest.mark.parametrize('name', ['concatenate', 'mean', 'asarray'])
    def test_signature_error_message_simple(self, name):
        func = getattr(np, name)
        try:
            func()
        except TypeError as e:
            exc = e
        assert exc.args[0].startswith(f'{name}()')

    def test_signature_error_message(self):

        def _dispatcher():
            return ()

        @array_function_dispatch(_dispatcher)
        def func():
            pass
        try:
            func._implementation(bad_arg=3)
        except TypeError as e:
            expected_exception = e
        try:
            func(bad_arg=3)
            raise AssertionError('must fail')
        except TypeError as exc:
            if exc.args[0].startswith('_dispatcher'):
                pytest.skip('Python version is not using __qualname__ for TypeError formatting.')
            assert exc.args == expected_exception.args

    @pytest.mark.parametrize('value', [234, 'this func is not replaced'])
    def test_dispatcher_error(self, value):
        error = TypeError(value)

        def dispatcher():
            raise error

        @array_function_dispatch(dispatcher)
        def func():
            return 3
        try:
            func()
            raise AssertionError('must fail')
        except TypeError as exc:
            assert exc is error

    def test_properties(self):
        func = dispatched_two_arg
        assert str(func) == str(func._implementation)
        repr_no_id = repr(func).split('at ')[0]
        repr_no_id_impl = repr(func._implementation).split('at ')[0]
        assert repr_no_id == repr_no_id_impl

    @pytest.mark.parametrize('func', [lambda x, y: 0, lambda like=None: 0, lambda *, like=None, a=3: 0])
    def test_bad_like_sig(self, func):
        with pytest.raises(RuntimeError):
            array_function_dispatch()(func)

    def test_bad_like_passing(self):

        def func(*, like=None):
            pass
        func_with_like = array_function_dispatch()(func)
        with pytest.raises(TypeError):
            func_with_like()
        with pytest.raises(TypeError):
            func_with_like(like=234)

    def test_too_many_args(self):
        objs = []
        for i in range(40):

            class MyArr:

                def __array_function__(self, *args, **kwargs):
                    return NotImplemented
            objs.append(MyArr())

        def _dispatch(*args):
            return args

        @array_function_dispatch(_dispatch)
        def func(*args):
            pass
        with pytest.raises(TypeError, match='maximum number'):
            func(*objs)