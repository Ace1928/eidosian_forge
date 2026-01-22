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
def _make_all_dtypes(no_float16, no_bool, no_complex):
    if no_float16:
        dtypes = _regular_float_dtypes
    else:
        dtypes = _float_dtypes
    if no_bool:
        dtypes += _int_dtypes
    else:
        dtypes += _int_bool_dtypes
    if not no_complex:
        dtypes += _complex_dtypes
    return dtypes