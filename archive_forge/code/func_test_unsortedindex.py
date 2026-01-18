import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
def test_unsortedindex():
    mi = MultiIndex.from_tuples([('z', 'a'), ('x', 'a'), ('y', 'b'), ('x', 'b'), ('y', 'a'), ('z', 'b')], names=['one', 'two'])
    df = DataFrame([[i, 10 * i] for i in range(6)], index=mi, columns=['one', 'two'])
    result = df.loc(axis=0)['z', 'a']
    expected = df.iloc[0]
    tm.assert_series_equal(result, expected)
    msg = 'MultiIndex slicing requires the index to be lexsorted: slicing on levels \\[1\\], lexsort depth 0'
    with pytest.raises(UnsortedIndexError, match=msg):
        df.loc(axis=0)['z', slice('a')]
    df.sort_index(inplace=True)
    assert len(df.loc(axis=0)['z', :]) == 2
    with pytest.raises(KeyError, match="'q'"):
        df.loc(axis=0)['q', :]