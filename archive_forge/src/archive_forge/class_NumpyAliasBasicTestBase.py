import contextlib
import inspect
from typing import Callable
import unittest
from unittest import mock
import warnings
import numpy
import cupy
from cupy._core import internal
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
class NumpyAliasBasicTestBase(NumpyAliasTestBase):

    def test_argspec(self):
        f = inspect.signature
        assert f(self.cupy_func) == f(self.numpy_func)

    def test_docstring(self):
        cupy_func = self.cupy_func
        numpy_func = self.numpy_func
        assert hasattr(cupy_func, '__doc__')
        assert cupy_func.__doc__ is not None
        assert cupy_func.__doc__ != ''
        assert cupy_func.__doc__ is not numpy_func.__doc__