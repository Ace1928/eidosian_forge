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
class TestArrayFunctionDispatch:

    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            roundtripped = pickle.loads(pickle.dumps(dispatched_one_arg, protocol=proto))
            assert_(roundtripped is dispatched_one_arg)

    def test_name_and_docstring(self):
        assert_equal(dispatched_one_arg.__name__, 'dispatched_one_arg')
        if sys.flags.optimize < 2:
            assert_equal(dispatched_one_arg.__doc__, 'Docstring.')

    def test_interface(self):

        class MyArray:

            def __array_function__(self, func, types, args, kwargs):
                return (self, func, types, args, kwargs)
        original = MyArray()
        obj, func, types, args, kwargs = dispatched_one_arg(original)
        assert_(obj is original)
        assert_(func is dispatched_one_arg)
        assert_equal(set(types), {MyArray})
        assert_(args == (original,))
        assert_equal(kwargs, {})

    def test_not_implemented(self):

        class MyArray:

            def __array_function__(self, func, types, args, kwargs):
                return NotImplemented
        array = MyArray()
        with assert_raises_regex(TypeError, 'no implementation found'):
            dispatched_one_arg(array)

    def test_where_dispatch(self):

        class DuckArray:

            def __array_function__(self, ufunc, method, *inputs, **kwargs):
                return 'overridden'
        array = np.array(1)
        duck_array = DuckArray()
        result = np.std(array, where=duck_array)
        assert_equal(result, 'overridden')