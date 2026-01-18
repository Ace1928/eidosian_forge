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
def test_partition_exception_out_of_range(self):
    pyfunc = partition
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    a = np.arange(10)

    def _check(a, kth):
        with self.assertRaises(ValueError) as e:
            cfunc(a, kth)
        assert str(e.exception) == 'kth out of bounds'
    _check(a, 10)
    _check(a, -11)
    _check(a, (3, 30))