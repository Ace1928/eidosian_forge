import functools
import inspect
import os
import random
from typing import Tuple, Type
import traceback
import unittest
import warnings
import numpy
import cupy
from cupy.testing import _array
from cupy.testing import _parameterized
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def _call_func_numpy_cupy(impl, args, kw, name, sp_name, scipy_name):
    cupy_result, cupy_error = _call_func_cupy(impl, args, kw, name, sp_name, scipy_name)
    numpy_result, numpy_error = _call_func_numpy(impl, args, kw, name, sp_name, scipy_name)
    return (cupy_result, cupy_error, numpy_result, numpy_error)