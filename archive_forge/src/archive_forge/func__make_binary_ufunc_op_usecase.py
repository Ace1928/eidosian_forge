import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def _make_binary_ufunc_op_usecase(ufunc_op):
    ldict = {}
    exec('def fn(x,y):\n    return x{0}y'.format(ufunc_op), globals(), ldict)
    fn = ldict['fn']
    fn.__name__ = 'usecase_{0}'.format(hash(ufunc_op))
    return fn