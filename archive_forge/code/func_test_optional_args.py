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