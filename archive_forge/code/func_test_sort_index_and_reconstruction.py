import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_and_reconstruction(self):
    df = DataFrame([[1, 1], [2, 2]], index=list('ab'))
    expected = DataFrame([[1, 1], [2, 2], [1, 1], [2, 2]], index=MultiIndex.from_tuples([(0.5, 'a'), (0.5, 'b'), (0.8, 'a'), (0.8, 'b')]))
    assert expected.index._is_lexsorted()
    result = DataFrame([[1, 1], [2, 2], [1, 1], [2, 2]], index=MultiIndex.from_product([[0.5, 0.8], list('ab')]))
    result = result.sort_index()
    assert result.index.is_monotonic_increasing
    tm.assert_frame_equal(result, expected)
    result = DataFrame([[1, 1], [2, 2], [1, 1], [2, 2]], index=MultiIndex(levels=[[0.5, 0.8], ['a', 'b']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]))
    result = result.sort_index()
    assert result.index._is_lexsorted()
    tm.assert_frame_equal(result, expected)
    concatted = pd.concat([df, df], keys=[0.8, 0.5])
    result = concatted.sort_index()
    assert result.index.is_monotonic_increasing
    tm.assert_frame_equal(result, expected)
    df = DataFrame([[1, 2], [6, 7]], columns=MultiIndex.from_tuples([(0, '20160811 12:00:00'), (0, '20160809 12:00:00')], names=['l1', 'Date']))
    df.columns = df.columns.set_levels(pd.to_datetime(df.columns.levels[1]), level=1)
    assert not df.columns.is_monotonic_increasing
    result = df.sort_index(axis=1)
    assert result.columns.is_monotonic_increasing
    result = df.sort_index(axis=1, level=1)
    assert result.columns.is_monotonic_increasing