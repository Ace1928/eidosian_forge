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
def test_np_trapz_basic(self):
    pyfunc = np_trapz
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc)
    y = [1, 2, 3]
    _check({'y': y})
    y = (3, 1, 2, 2, 2)
    _check({'y': y})
    y = np.arange(15).reshape(3, 5)
    _check({'y': y})
    y = np.linspace(-10, 10, 60).reshape(4, 3, 5)
    _check({'y': y}, abs_tol=1e-13)
    self.rnd.shuffle(y)
    _check({'y': y}, abs_tol=1e-13)
    y = np.array([])
    _check({'y': y})
    y = np.array([3.142, np.nan, np.inf, -np.inf, 5])
    _check({'y': y})
    y = np.arange(20) + np.linspace(0, 10, 20) * 1j
    _check({'y': y})
    y = np.array([], dtype=np.complex128)
    _check({'y': y})
    y = (True, False, True)
    _check({'y': y})