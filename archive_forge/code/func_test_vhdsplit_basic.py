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
def test_vhdsplit_basic(self):

    def inputs1D():
        yield (np.array([1, 2, 3, 4]), 2)
        yield (np.array([1.0, 2.0, 3.0, 4.0]), 2)

    def inputs2D():
        yield (np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), 2)
        yield (np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]), 2)
        yield (np.arange(16.0).reshape(4, 4), 2)
        yield (np.arange(16.0).reshape(4, 4), np.array([3, 6]))
        yield (np.arange(16.0).reshape(4, 4), [3, 6])
        yield (np.arange(16.0).reshape(4, 4), (3, 6))
        yield (np.arange(8.0).reshape(2, 2, 2), 2)

    def inputs3D():
        (np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]]), 2)
        yield (np.arange(16.0).reshape(2, 2, 4), 2)
        yield (np.arange(16.0).reshape(2, 2, 4), np.array([3, 6]))
        yield (np.arange(16.0).reshape(2, 2, 4), [3, 6])
        yield (np.arange(16.0).reshape(2, 2, 4), (3, 6))
        yield (np.arange(8.0).reshape(2, 2, 2), 2)
    inputs = [inputs1D(), inputs2D(), inputs3D()]
    for f, mindim, name in [(vsplit, 2, 'vsplit'), (hsplit, 1, 'hsplit'), (dsplit, 3, 'dsplit')]:
        pyfunc = f
        cfunc = njit(pyfunc)
        for i in range(mindim, 4):
            for a, ind_or_sec in inputs[i - 1]:
                self.assertPreciseEqual(pyfunc(a, ind_or_sec), cfunc(a, ind_or_sec))