import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('vc_sort', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
def test_value_counts_sort(sort, vc_sort, normalize):
    df = DataFrame({'a': [2, 1, 1, 1], 0: [3, 4, 3, 3]})
    gb = df.groupby('a', sort=sort)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)
    if normalize:
        values = [2 / 3, 1 / 3, 1.0]
    else:
        values = [2, 1, 1]
    index = MultiIndex(levels=[[1, 2], [3, 4]], codes=[[0, 0, 1], [0, 1, 0]], names=['a', 0])
    expected = Series(values, index=index, name='proportion' if normalize else 'count')
    if sort and vc_sort:
        taker = [0, 1, 2]
    elif sort and (not vc_sort):
        taker = [0, 1, 2]
    elif not sort and vc_sort:
        taker = [0, 2, 1]
    else:
        taker = [2, 1, 0]
    expected = expected.take(taker)
    tm.assert_series_equal(result, expected)