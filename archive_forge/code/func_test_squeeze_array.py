from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_squeeze_array(self, flags=enable_pyobj_flags):
    a = np.arange(2 * 1 * 3 * 1 * 4).reshape(2, 1, 3, 1, 4)
    pyfunc = squeeze_array
    arraytype1 = typeof(a)
    cfunc = jit((arraytype1,), **flags)(pyfunc)
    expected = pyfunc(a)
    got = cfunc(a)
    np.testing.assert_equal(expected, got)