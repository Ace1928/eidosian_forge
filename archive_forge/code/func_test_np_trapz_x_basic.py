import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_np_trapz_x_basic(self):
    pyfunc = np_trapz_x
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc)
    y = [1, 2, 3]
    x = [4, 6, 8]
    _check({'y': y, 'x': x})
    y = [1, 2, 3, 4, 5]
    x = (4, 6)
    _check({'y': y, 'x': x})
    y = (1, 2, 3, 4, 5)
    x = [4, 5, 6, 7, 8]
    _check({'y': y, 'x': x})
    y = np.array([1, 2, 3, 4, 5])
    x = [4, 4]
    _check({'y': y, 'x': x})
    y = np.array([])
    x = np.array([2, 3])
    _check({'y': y, 'x': x})
    y = (1, 2, 3, 4, 5)
    x = None
    _check({'y': y, 'x': x})
    y = np.arange(20).reshape(5, 4)
    x = np.array([4, 5])
    _check({'y': y, 'x': x})
    y = np.arange(20).reshape(5, 4)
    x = np.array([4, 5, 6, 7])
    _check({'y': y, 'x': x})
    y = np.arange(60).reshape(5, 4, 3)
    x = np.array([4, 5])
    _check({'y': y, 'x': x})
    y = np.arange(60).reshape(5, 4, 3)
    x = np.array([4, 5, 7])
    _check({'y': y, 'x': x})
    y = np.arange(60).reshape(5, 4, 3)
    self.rnd.shuffle(y)
    x = y + 1.1
    self.rnd.shuffle(x)
    _check({'y': y, 'x': x})
    y = np.arange(20)
    x = y + np.linspace(0, 10, 20) * 1j
    _check({'y': y, 'x': x})
    y = np.array([1, 2, 3])
    x = np.array([1 + 1j, 1 + 2j])
    _check({'y': y, 'x': x})