from functools import partial
import itertools
from itertools import chain, product, starmap
import sys
import numpy as np
from numba import jit, literally, njit, typeof, TypingError
from numba.core import utils, types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.types.functions import _header_lead
import unittest
def test_slice_indices(self):
    """Test that a numba slice returns same result for .indices as a python one."""
    slices = starmap(slice, product(chain(range(-5, 5), (None,)), chain(range(-5, 5), (None,)), chain(range(-5, 5), (None,))))
    lengths = range(-2, 3)
    cfunc = jit(nopython=True)(slice_indices)
    for s, l in product(slices, lengths):
        try:
            expected = slice_indices(s, l)
        except Exception as py_e:
            with self.assertRaises(type(py_e)) as numba_e:
                cfunc(s, l)
            self.assertIn(str(py_e), str(numba_e.exception))
        else:
            self.assertPreciseEqual(expected, cfunc(s, l))