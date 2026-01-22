from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def _val_variations():
    yield 1
    yield 3.142
    yield np.nan
    yield (-np.inf)
    yield True
    yield np.arange(4)
    yield (4,)
    yield [8, 9]
    yield np.arange(54).reshape(9, 3, 2, 1)
    yield np.asfortranarray(np.arange(9).reshape(3, 3))
    yield np.arange(9).reshape(3, 3)[::-1]