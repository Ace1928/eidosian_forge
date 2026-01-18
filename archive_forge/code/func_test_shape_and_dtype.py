import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_shape_and_dtype(self):
    sizes = (4, 5, 3, 2)
    for func in (range, np.arange):
        arrays = np.ix_(*[func(sz) for sz in sizes])
        for k, (a, sz) in enumerate(zip(arrays, sizes)):
            assert_equal(a.shape[k], sz)
            assert_(all((sh == 1 for j, sh in enumerate(a.shape) if j != k)))
            assert_(np.issubdtype(a.dtype, np.integer))