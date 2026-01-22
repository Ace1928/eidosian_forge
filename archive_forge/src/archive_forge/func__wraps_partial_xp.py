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
def _wraps_partial_xp(wrapped, name, sp_name, scipy_name):
    names = [name, sp_name, scipy_name]
    names = [n for n in names if n is not None]
    return _wraps_partial(wrapped, *names)