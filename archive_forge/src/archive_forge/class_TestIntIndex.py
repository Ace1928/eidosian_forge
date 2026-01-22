import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
class TestIntIndex:

    def test_check_integrity(self):
        msg = 'Too many indices'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=1, indices=[1, 2, 3])
        msg = 'No index can be less than zero'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])
        msg = 'No index can be less than zero'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])
        msg = 'All indices must be less than the length'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 5])
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 6])
        msg = 'Indices must be strictly increasing'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 2])
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 3])

    def test_int_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind='integer')
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind='integer')
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))
        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind='integer')
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_equals(self):
        index = IntIndex(10, [0, 1, 2, 3, 4])
        assert index.equals(index)
        assert not index.equals(IntIndex(10, [0, 1, 2, 3]))

    def test_to_block_index(self, cases, test_length):
        xloc, xlen, yloc, ylen, _, _ = cases
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)
        xbindex = xindex.to_int_index().to_block_index()
        ybindex = yindex.to_int_index().to_block_index()
        assert isinstance(xbindex, BlockIndex)
        assert xbindex.equals(xindex)
        assert ybindex.equals(yindex)

    def test_to_int_index(self):
        index = IntIndex(10, [2, 3, 4, 5, 6])
        assert index.to_int_index() is index