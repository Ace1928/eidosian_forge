import numpy as np
import pytest
from pandas import IntervalIndex
import pandas._testing as tm
from pandas.tests.indexes.common import Base
def test_where(self, simple_index, listlike_box):
    klass = listlike_box
    idx = simple_index
    cond = [True] * len(idx)
    expected = idx
    result = expected.where(klass(cond))
    tm.assert_index_equal(result, expected)
    cond = [False] + [True] * len(idx[1:])
    expected = IntervalIndex([np.nan] + idx[1:].tolist())
    result = idx.where(klass(cond))
    tm.assert_index_equal(result, expected)