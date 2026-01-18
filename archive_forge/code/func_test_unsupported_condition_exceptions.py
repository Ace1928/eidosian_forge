import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def test_unsupported_condition_exceptions(self):
    err_idx_cases = [('Multi-dimensional indices are not supported.', (0, 3, np.array([[1, 2], [2, 3]]))), ('Using more than one non-scalar array index is unsupported.', (0, 3, np.array([1, 2]), np.array([1, 2]))), ('Using more than one indexing subspace is unsupported.' + ' An indexing subspace is a group of one or more consecutive' + ' indices comprising integer or array types.', (0, np.array([1, 2]), slice(None), 3, 4))]
    for err, idx in err_idx_cases:
        with self.assertRaises(TypingError) as raises:
            self.check_getitem_indices(self.shape, idx)
        self.assertIn(err, str(raises.exception))