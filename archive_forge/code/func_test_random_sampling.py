import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_random_sampling(self):
    for f in (sprand, _sprandn):
        for t in [np.float32, np.float64, np.longdouble, np.int32, np.int64, np.complex64, np.complex128]:
            x = f(5, 10, density=0.1, dtype=t)
            assert_equal(x.dtype, t)
            assert_equal(x.shape, (5, 10))
            assert_equal(x.nnz, 5)
        x1 = f(5, 10, density=0.1, random_state=4321)
        assert_equal(x1.dtype, np.float64)
        x2 = f(5, 10, density=0.1, random_state=np.random.RandomState(4321))
        assert_array_equal(x1.data, x2.data)
        assert_array_equal(x1.row, x2.row)
        assert_array_equal(x1.col, x2.col)
        for density in [0.0, 0.1, 0.5, 1.0]:
            x = f(5, 10, density=density)
            assert_equal(x.nnz, int(density * np.prod(x.shape)))
        for fmt in ['coo', 'csc', 'csr', 'lil']:
            x = f(5, 10, format=fmt)
            assert_equal(x.format, fmt)
        assert_raises(ValueError, lambda: f(5, 10, 1.1))
        assert_raises(ValueError, lambda: f(5, 10, -0.1))