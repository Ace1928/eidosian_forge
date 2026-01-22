from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def _multi_dimensional_array_variations_strided(n):
    for shape in _shape_variations(n):
        tmp = np.zeros(tuple([x * 2 for x in shape]), dtype=np.float64)
        slicer = tuple((slice(0, x * 2, 2) for x in shape))
        yield tmp[slicer]