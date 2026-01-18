from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
@njit
def transpose_issue_4708(m, n):
    r1 = np.reshape(np.arange(m * n * 3), (m, 3, n))
    r2 = np.reshape(np.arange(n * 3), (n, 3))
    r_dif = (r1 - r2.T).T
    r_dif = np.transpose(r_dif, (2, 0, 1))
    z = r_dif + 1
    return z