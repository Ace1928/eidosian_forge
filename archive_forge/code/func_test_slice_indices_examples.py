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
def test_slice_indices_examples(self):
    """Tests for specific error cases."""
    cslice_indices = jit(nopython=True)(slice_indices)
    with self.assertRaises(TypingError) as e:
        cslice_indices(slice(None), 1, 2, 3)
    self.assertIn('indices() takes exactly one argument (3 given)', str(e.exception))
    with self.assertRaises(TypingError) as e:
        cslice_indices(slice(None, None, 0), 1.2)
    self.assertIn("'%s' object cannot be interpreted as an integer" % typeof(1.2), str(e.exception))