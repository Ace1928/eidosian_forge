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
def _wraps_partial(wrapped, *names):

    def decorator(impl):
        impl = functools.wraps(wrapped)(impl)
        impl.__signature__ = inspect.signature(functools.partial(wrapped, **{name: None for name in names}))
        return impl
    return decorator