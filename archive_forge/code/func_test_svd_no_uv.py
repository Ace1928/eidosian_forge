import warnings
import numpy as np
from numpy import linalg, arange, float64, array, dot, transpose
from numpy.testing import (
def test_svd_no_uv(self):
    for shape in ((3, 4), (4, 4), (4, 3)):
        for t in (float, complex):
            a = np.ones(shape, dtype=t)
            w = linalg.svd(a, compute_uv=False)
            c = np.count_nonzero(np.absolute(w) > 0.5)
            assert_equal(c, 1)
            assert_equal(np.linalg.matrix_rank(a), 1)
            assert_array_less(1, np.linalg.norm(a, ord=2))