import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
class TestBlockIndex:

    def test_block_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind='block')
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind='block')
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))
        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind='block')
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))
        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind='block')
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    @pytest.mark.parametrize('i', [5, 10, 100, 101])
    def test_make_block_boundary(self, i):
        idx = make_sparse_index(i, np.arange(0, i, 2, dtype=np.int32), kind='block')
        exp = np.arange(0, i, 2, dtype=np.int32)
        tm.assert_numpy_array_equal(idx.blocs, exp)
        tm.assert_numpy_array_equal(idx.blengths, np.ones(len(exp), dtype=np.int32))

    def test_equals(self):
        index = BlockIndex(10, [0, 4], [2, 5])
        assert index.equals(index)
        assert not index.equals(BlockIndex(10, [0, 4], [2, 6]))

    def test_check_integrity(self):
        locs = []
        lengths = []
        BlockIndex(0, locs, lengths)
        BlockIndex(1, locs, lengths)
        msg = 'Block 0 extends beyond end'
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [5], [10])
        msg = 'Block 0 overlaps'
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [2, 5], [5, 3])

    def test_to_int_index(self):
        locs = [0, 10]
        lengths = [4, 6]
        exp_inds = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]
        block = BlockIndex(20, locs, lengths)
        dense = block.to_int_index()
        tm.assert_numpy_array_equal(dense.indices, np.array(exp_inds, dtype=np.int32))

    def test_to_block_index(self):
        index = BlockIndex(10, [0, 5], [4, 5])
        assert index.to_block_index() is index