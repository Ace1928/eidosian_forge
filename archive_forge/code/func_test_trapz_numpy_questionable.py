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
@unittest.skip('NumPy behaviour questionable')
def test_trapz_numpy_questionable(self):
    pyfunc = np_trapz
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc)
    y = np.array([True, False, True, True]).astype(int)
    _check({'y': y})
    y = np.array([True, False, True, True])
    _check({'y': y})