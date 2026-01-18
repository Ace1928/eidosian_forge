import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_union_empty_result(self, closed, sort):
    index = empty_index(dtype='int64', closed=closed)
    result = index.union(index, sort=sort)
    tm.assert_index_equal(result, index)
    other = empty_index(dtype='float64', closed=closed)
    result = index.union(other, sort=sort)
    expected = other
    tm.assert_index_equal(result, expected)
    other = index.union(index, sort=sort)
    tm.assert_index_equal(result, expected)
    other = empty_index(dtype='uint64', closed=closed)
    result = index.union(other, sort=sort)
    tm.assert_index_equal(result, expected)
    result = other.union(index, sort=sort)
    tm.assert_index_equal(result, expected)