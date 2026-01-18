import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_tuple_types_exception(self):
    with self.assertRaises(errors.TypingError) as raises:
        types.Tuple(types.uint32)
    self.assertIn("Argument 'types' is not iterable", str(raises.exception))