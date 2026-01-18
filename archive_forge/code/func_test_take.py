import numpy as np
import pytest
from pandas import IntervalIndex
import pandas._testing as tm
from pandas.tests.indexes.common import Base
def test_take(self, closed):
    index = self.create_index(closed=closed)
    result = index.take(range(10))
    tm.assert_index_equal(result, index)
    result = index.take([0, 0, 1])
    expected = IntervalIndex.from_arrays([0, 0, 1], [1, 1, 2], closed=closed)
    tm.assert_index_equal(result, expected)