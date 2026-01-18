import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_partial_set(self):
    ser = Series([0.1, 0.2], index=[1, 2])
    expected = Series([np.nan, 0.2, np.nan], index=[3, 2, 3])
    with pytest.raises(KeyError, match='not in index'):
        ser.loc[[3, 2, 3]]
    result = ser.reindex([3, 2, 3])
    tm.assert_series_equal(result, expected, check_index_type=True)
    expected = Series([np.nan, 0.2, np.nan, np.nan], index=[3, 2, 3, 'x'])
    with pytest.raises(KeyError, match='not in index'):
        ser.loc[[3, 2, 3, 'x']]
    result = ser.reindex([3, 2, 3, 'x'])
    tm.assert_series_equal(result, expected, check_index_type=True)
    expected = Series([0.2, 0.2, 0.1], index=[2, 2, 1])
    result = ser.loc[[2, 2, 1]]
    tm.assert_series_equal(result, expected, check_index_type=True)
    expected = Series([0.2, 0.2, np.nan, 0.1], index=[2, 2, 'x', 1])
    with pytest.raises(KeyError, match='not in index'):
        ser.loc[[2, 2, 'x', 1]]
    result = ser.reindex([2, 2, 'x', 1])
    tm.assert_series_equal(result, expected, check_index_type=True)
    msg = f'''\\"None of \\[Index\\(\\[3, 3, 3\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
    with pytest.raises(KeyError, match=msg):
        ser.loc[[3, 3, 3]]
    expected = Series([0.2, 0.2, np.nan], index=[2, 2, 3])
    with pytest.raises(KeyError, match='not in index'):
        ser.loc[[2, 2, 3]]
    result = ser.reindex([2, 2, 3])
    tm.assert_series_equal(result, expected, check_index_type=True)
    s = Series([0.1, 0.2, 0.3], index=[1, 2, 3])
    expected = Series([0.3, np.nan, np.nan], index=[3, 4, 4])
    with pytest.raises(KeyError, match='not in index'):
        s.loc[[3, 4, 4]]
    result = s.reindex([3, 4, 4])
    tm.assert_series_equal(result, expected, check_index_type=True)
    s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
    expected = Series([np.nan, 0.3, 0.3], index=[5, 3, 3])
    with pytest.raises(KeyError, match='not in index'):
        s.loc[[5, 3, 3]]
    result = s.reindex([5, 3, 3])
    tm.assert_series_equal(result, expected, check_index_type=True)
    s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
    expected = Series([np.nan, 0.4, 0.4], index=[5, 4, 4])
    with pytest.raises(KeyError, match='not in index'):
        s.loc[[5, 4, 4]]
    result = s.reindex([5, 4, 4])
    tm.assert_series_equal(result, expected, check_index_type=True)
    s = Series([0.1, 0.2, 0.3, 0.4], index=[4, 5, 6, 7])
    expected = Series([0.4, np.nan, np.nan], index=[7, 2, 2])
    with pytest.raises(KeyError, match='not in index'):
        s.loc[[7, 2, 2]]
    result = s.reindex([7, 2, 2])
    tm.assert_series_equal(result, expected, check_index_type=True)
    s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
    expected = Series([0.4, np.nan, np.nan], index=[4, 5, 5])
    with pytest.raises(KeyError, match='not in index'):
        s.loc[[4, 5, 5]]
    result = s.reindex([4, 5, 5])
    tm.assert_series_equal(result, expected, check_index_type=True)
    expected = Series([0.2, 0.2, 0.1, 0.1], index=[2, 2, 1, 1])
    result = ser.iloc[[1, 1, 0, 0]]
    tm.assert_series_equal(result, expected, check_index_type=True)