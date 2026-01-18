import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
@pytest.mark.parametrize('level0_value', ['D', 'A'])
def test_series_getitem_duplicates_multiindex(level0_value):
    index = MultiIndex(levels=[[level0_value, 'B', 'C'], [0, 26, 27, 37, 57, 67, 75, 82]], codes=[[0, 0, 0, 1, 2, 2, 2, 2, 2, 2], [1, 3, 4, 6, 0, 2, 2, 3, 5, 7]], names=['tag', 'day'])
    arr = np.random.default_rng(2).standard_normal((len(index), 1))
    df = DataFrame(arr, index=index, columns=['val'])
    if level0_value != 'A':
        with pytest.raises(KeyError, match="^'A'$"):
            df.val['A']
    with pytest.raises(KeyError, match="^'X'$"):
        df.val['X']
    result = df.val[level0_value]
    expected = Series(arr.ravel()[0:3], name='val', index=Index([26, 37, 57], name='day'))
    tm.assert_series_equal(result, expected)