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
def test_vander_basic(self):
    pyfunc = vander
    cfunc = jit(nopython=True)(pyfunc)
    _check_output = partial(self._check_output, pyfunc, cfunc)

    def _check(x):
        n_choices = [None, 0, 1, 2, 3, 4]
        increasing_choices = [True, False]
        params = {'x': x}
        _check_output(params)
        for n in n_choices:
            params = {'x': x, 'N': n}
            _check_output(params)
        for increasing in increasing_choices:
            params = {'x': x, 'increasing': increasing}
            _check_output(params)
        for n in n_choices:
            for increasing in increasing_choices:
                params = {'x': x, 'N': n, 'increasing': increasing}
                _check_output(params)
    _check(np.array([1, 2, 3, 5]))
    _check(np.arange(7) - 10.5)
    _check(np.linspace(3, 10, 5))
    _check(np.array([1.2, np.nan, np.inf, -np.inf]))
    _check(np.array([]))
    _check(np.arange(-5, 5) - 0.3)
    _check(np.array([True] * 5 + [False] * 4))
    for dtype in (np.int32, np.int64, np.float32, np.float64):
        _check(np.arange(10, dtype=dtype))
    _check([0, 1, 2, 3])
    _check((4, 5, 6, 7))
    _check((0.0, 1.0, 2.0))
    _check(())
    _check((3, 4.444, 3.142))
    _check((True, False, 4))