from datetime import (
import numpy as np
import pytest
from pandas.errors import UnsortedIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
def test_per_axis_per_level_setitem(self):
    idx = pd.IndexSlice
    index = MultiIndex.from_tuples([('A', 1), ('A', 2), ('A', 3), ('B', 1)], names=['one', 'two'])
    columns = MultiIndex.from_tuples([('a', 'foo'), ('a', 'bar'), ('b', 'foo'), ('b', 'bah')], names=['lvl0', 'lvl1'])
    df_orig = DataFrame(np.arange(16, dtype='int64').reshape(4, 4), index=index, columns=columns)
    df_orig = df_orig.sort_index(axis=0).sort_index(axis=1)
    df = df_orig.copy()
    df.loc[(slice(None), slice(None)), :] = 100
    expected = df_orig.copy()
    expected.iloc[:, :] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc(axis=0)[:, :] = 100
    expected = df_orig.copy()
    expected.iloc[:, :] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[(slice(None), slice(None)), (slice(None), slice(None))] = 100
    expected = df_orig.copy()
    expected.iloc[:, :] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[:, (slice(None), slice(None))] = 100
    expected = df_orig.copy()
    expected.iloc[:, :] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[(slice(None), [1]), :] = 100
    expected = df_orig.copy()
    expected.iloc[[0, 3]] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[(slice(None), 1), :] = 100
    expected = df_orig.copy()
    expected.iloc[[0, 3]] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc(axis=0)[:, 1] = 100
    expected = df_orig.copy()
    expected.iloc[[0, 3]] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[:, (slice(None), ['foo'])] = 100
    expected = df_orig.copy()
    expected.iloc[:, [1, 3]] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[(slice(None), 1), (slice(None), ['foo'])] = 100
    expected = df_orig.copy()
    expected.iloc[[0, 3], [1, 3]] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[idx[:, 1], idx[:, ['foo']]] = 100
    expected = df_orig.copy()
    expected.iloc[[0, 3], [1, 3]] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc['A', 'a'] = 100
    expected = df_orig.copy()
    expected.iloc[0:3, 0:2] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[(slice(None), 1), (slice(None), ['foo'])] = np.array([[100, 100], [100, 100]], dtype='int64')
    expected = df_orig.copy()
    expected.iloc[[0, 3], [1, 3]] = 100
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    msg = 'setting an array element with a sequence.'
    with pytest.raises(ValueError, match=msg):
        df.loc[(slice(None), 1), (slice(None), ['foo'])] = np.array([[100], [100, 100]], dtype='int64')
    msg = 'Must have equal len keys and value when setting with an iterable'
    with pytest.raises(ValueError, match=msg):
        df.loc[(slice(None), 1), (slice(None), ['foo'])] = np.array([100, 100, 100, 100], dtype='int64')
    df = df_orig.copy()
    df.loc[(slice(None), 1), (slice(None), ['foo'])] = df.loc[(slice(None), 1), (slice(None), ['foo'])] * 5
    expected = df_orig.copy()
    expected.iloc[[0, 3], [1, 3]] = expected.iloc[[0, 3], [1, 3]] * 5
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[(slice(None), 1), (slice(None), ['foo'])] *= df.loc[(slice(None), 1), (slice(None), ['foo'])]
    expected = df_orig.copy()
    expected.iloc[[0, 3], [1, 3]] *= expected.iloc[[0, 3], [1, 3]]
    tm.assert_frame_equal(df, expected)
    rhs = df_orig.loc[(slice(None), 1), (slice(None), ['foo'])].copy()
    rhs.loc[:, ('c', 'bah')] = 10
    df = df_orig.copy()
    df.loc[(slice(None), 1), (slice(None), ['foo'])] *= rhs
    expected = df_orig.copy()
    expected.iloc[[0, 3], [1, 3]] *= expected.iloc[[0, 3], [1, 3]]
    tm.assert_frame_equal(df, expected)