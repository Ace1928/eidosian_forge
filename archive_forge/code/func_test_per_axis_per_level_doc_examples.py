from datetime import (
import numpy as np
import pytest
from pandas.errors import UnsortedIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
def test_per_axis_per_level_doc_examples(self):
    idx = pd.IndexSlice
    index = MultiIndex.from_product([_mklbl('A', 4), _mklbl('B', 2), _mklbl('C', 4), _mklbl('D', 2)])
    columns = MultiIndex.from_tuples([('a', 'foo'), ('a', 'bar'), ('b', 'foo'), ('b', 'bah')], names=['lvl0', 'lvl1'])
    df = DataFrame(np.arange(len(index) * len(columns), dtype='int64').reshape((len(index), len(columns))), index=index, columns=columns)
    result = df.loc[(slice('A1', 'A3'), slice(None), ['C1', 'C3']), :]
    expected = df.loc[[(a, b, c, d) for a, b, c, d in df.index.values if a in ('A1', 'A2', 'A3') and c in ('C1', 'C3')]]
    tm.assert_frame_equal(result, expected)
    result = df.loc[idx['A1':'A3', :, ['C1', 'C3']], :]
    tm.assert_frame_equal(result, expected)
    result = df.loc[(slice(None), slice(None), ['C1', 'C3']), :]
    expected = df.loc[[(a, b, c, d) for a, b, c, d in df.index.values if c in ('C1', 'C3')]]
    tm.assert_frame_equal(result, expected)
    result = df.loc[idx[:, :, ['C1', 'C3']], :]
    tm.assert_frame_equal(result, expected)
    msg = 'MultiIndex slicing requires the index to be lexsorted: slicing on levels \\[1\\], lexsort depth 1'
    with pytest.raises(UnsortedIndexError, match=msg):
        df.loc['A1', ('a', slice('foo'))]
    tm.assert_frame_equal(df.loc['A1', (slice(None), 'foo')], df.loc['A1'].iloc[:, [0, 2]])
    df = df.sort_index(axis=1)
    df.loc['A1', (slice(None), 'foo')]
    df.loc[(slice(None), slice(None), ['C1', 'C3']), (slice(None), 'foo')]
    df.loc(axis=0)[:, :, ['C1', 'C3']] = -10