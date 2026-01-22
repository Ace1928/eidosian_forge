import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
class TestSparseIndexIntersect:

    @td.skip_if_windows
    def test_intersect(self, cases, test_length):
        xloc, xlen, yloc, ylen, eloc, elen = cases
        xindex = BlockIndex(test_length, xloc, xlen)
        yindex = BlockIndex(test_length, yloc, ylen)
        expected = BlockIndex(test_length, eloc, elen)
        longer_index = BlockIndex(test_length + 1, yloc, ylen)
        result = xindex.intersect(yindex)
        assert result.equals(expected)
        result = xindex.to_int_index().intersect(yindex.to_int_index())
        assert result.equals(expected.to_int_index())
        msg = 'Indices must reference same underlying length'
        with pytest.raises(Exception, match=msg):
            xindex.intersect(longer_index)
        with pytest.raises(Exception, match=msg):
            xindex.to_int_index().intersect(longer_index.to_int_index())

    def test_intersect_empty(self):
        xindex = IntIndex(4, np.array([], dtype=np.int32))
        yindex = IntIndex(4, np.array([2, 3], dtype=np.int32))
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)
        xindex = xindex.to_block_index()
        yindex = yindex.to_block_index()
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

    @pytest.mark.parametrize('case', [IntIndex(5, np.array([1, 2], dtype=np.int32)), IntIndex(5, np.array([0, 2, 4], dtype=np.int32)), IntIndex(0, np.array([], dtype=np.int32)), IntIndex(5, np.array([], dtype=np.int32))])
    def test_intersect_identical(self, case):
        assert case.intersect(case).equals(case)
        case = case.to_block_index()
        assert case.intersect(case).equals(case)