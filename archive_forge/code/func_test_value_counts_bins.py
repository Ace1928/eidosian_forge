import collections
from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
def test_value_counts_bins(index_or_series, using_infer_string):
    klass = index_or_series
    s_values = ['a', 'b', 'b', 'b', 'b', 'c', 'd', 'd', 'a', 'a']
    s = klass(s_values)
    msg = 'bins argument only works with numeric data'
    with pytest.raises(TypeError, match=msg):
        s.value_counts(bins=1)
    s1 = Series([1, 1, 2, 3])
    res1 = s1.value_counts(bins=1)
    exp1 = Series({Interval(0.997, 3.0): 4}, name='count')
    tm.assert_series_equal(res1, exp1)
    res1n = s1.value_counts(bins=1, normalize=True)
    exp1n = Series({Interval(0.997, 3.0): 1.0}, name='proportion')
    tm.assert_series_equal(res1n, exp1n)
    if isinstance(s1, Index):
        tm.assert_index_equal(s1.unique(), Index([1, 2, 3]))
    else:
        exp = np.array([1, 2, 3], dtype=np.int64)
        tm.assert_numpy_array_equal(s1.unique(), exp)
    assert s1.nunique() == 3
    res4 = s1.value_counts(bins=4, dropna=True)
    intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
    exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 1, 3, 2]), name='count')
    tm.assert_series_equal(res4, exp4)
    res4 = s1.value_counts(bins=4, dropna=False)
    intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
    exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 1, 3, 2]), name='count')
    tm.assert_series_equal(res4, exp4)
    res4n = s1.value_counts(bins=4, normalize=True)
    exp4n = Series([0.5, 0.25, 0.25, 0], index=intervals.take([0, 1, 3, 2]), name='proportion')
    tm.assert_series_equal(res4n, exp4n)
    s_values = ['a', 'b', 'b', 'b', np.nan, np.nan, 'd', 'd', 'a', 'a', 'b']
    s = klass(s_values)
    expected = Series([4, 3, 2], index=['b', 'a', 'd'], name='count')
    tm.assert_series_equal(s.value_counts(), expected)
    if isinstance(s, Index):
        exp = Index(['a', 'b', np.nan, 'd'])
        tm.assert_index_equal(s.unique(), exp)
    else:
        exp = np.array(['a', 'b', np.nan, 'd'], dtype=object)
        if using_infer_string:
            exp = array(exp)
        tm.assert_equal(s.unique(), exp)
    assert s.nunique() == 3
    s = klass({}) if klass is dict else klass({}, dtype=object)
    expected = Series([], dtype=np.int64, name='count')
    tm.assert_series_equal(s.value_counts(), expected, check_index_type=False)
    if isinstance(s, Index):
        tm.assert_index_equal(s.unique(), Index([]), exact=False)
    else:
        tm.assert_numpy_array_equal(s.unique(), np.array([]), check_dtype=False)
    assert s.nunique() == 0