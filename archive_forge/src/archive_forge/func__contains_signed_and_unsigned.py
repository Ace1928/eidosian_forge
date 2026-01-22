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
def _contains_signed_and_unsigned(kw):

    def isdtype(v):
        if isinstance(v, numpy.dtype):
            return True
        elif isinstance(v, str):
            return True
        elif isinstance(v, type) and issubclass(v, numpy.number):
            return True
        else:
            return False
    vs = set((v for v in kw.values() if isdtype(v)))
    return any((d in vs for d in _unsigned_dtypes)) and any((d in vs for d in _float_dtypes + _signed_dtypes))