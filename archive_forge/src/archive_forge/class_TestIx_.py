import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
class TestIx_:

    def test_regression_1(self):
        a, = np.ix_(range(0))
        assert_equal(a.dtype, np.intp)
        a, = np.ix_([])
        assert_equal(a.dtype, np.intp)
        a, = np.ix_(np.array([], dtype=np.float32))
        assert_equal(a.dtype, np.float32)

    def test_shape_and_dtype(self):
        sizes = (4, 5, 3, 2)
        for func in (range, np.arange):
            arrays = np.ix_(*[func(sz) for sz in sizes])
            for k, (a, sz) in enumerate(zip(arrays, sizes)):
                assert_equal(a.shape[k], sz)
                assert_(all((sh == 1 for j, sh in enumerate(a.shape) if j != k)))
                assert_(np.issubdtype(a.dtype, np.integer))

    def test_bool(self):
        bool_a = [True, False, True, True]
        int_a, = np.nonzero(bool_a)
        assert_equal(np.ix_(bool_a)[0], int_a)

    def test_1d_only(self):
        idx2d = [[1, 2, 3], [4, 5, 6]]
        assert_raises(ValueError, np.ix_, idx2d)

    def test_repeated_input(self):
        length_of_vector = 5
        x = np.arange(length_of_vector)
        out = ix_(x, x)
        assert_equal(out[0].shape, (length_of_vector, 1))
        assert_equal(out[1].shape, (1, length_of_vector))
        assert_equal(x.shape, (length_of_vector,))