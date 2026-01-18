import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
@pytest.mark.parametrize('access_method', [lambda s, x: s[:, x], lambda s, x: s.loc[:, x], lambda s, x: s.xs(x, level=1)])
@pytest.mark.parametrize('level1_value, expected', [(0, Series([1], index=[0])), (1, Series([2, 3], index=[1, 2]))])
def test_series_getitem_multiindex(access_method, level1_value, expected):
    mi = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 1)], names=['A', 'B'])
    ser = Series([1, 2, 3], index=mi)
    expected.index.name = 'A'
    result = access_method(ser, level1_value)
    tm.assert_series_equal(result, expected)