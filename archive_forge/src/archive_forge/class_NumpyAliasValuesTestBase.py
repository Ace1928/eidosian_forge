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
class NumpyAliasValuesTestBase(NumpyAliasTestBase):

    def test_values(self):
        assert self.cupy_func(*self.args) == self.numpy_func(*self.args)