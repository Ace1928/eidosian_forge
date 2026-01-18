import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('f', [lambda x, y: x.expanding().cov(y), lambda x, y: x.expanding().corr(y), lambda x, y: x.rolling(window=3).cov(y), lambda x, y: x.rolling(window=3).corr(y), lambda x, y: x.ewm(com=3).cov(y), lambda x, y: x.ewm(com=3).corr(y)])
def test_pairwise_with_series(self, pairwise_frames, pairwise_target_frame, f):
    result = f(pairwise_frames, Series([1, 1, 3, 8]))
    tm.assert_index_equal(result.index, pairwise_frames.index)
    tm.assert_index_equal(result.columns, pairwise_frames.columns)
    expected = f(pairwise_target_frame, Series([1, 1, 3, 8]))
    result = result.dropna().values
    expected = expected.dropna().values
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)
    result = f(Series([1, 1, 3, 8]), pairwise_frames)
    tm.assert_index_equal(result.index, pairwise_frames.index)
    tm.assert_index_equal(result.columns, pairwise_frames.columns)
    expected = f(Series([1, 1, 3, 8]), pairwise_target_frame)
    result = result.dropna().values
    expected = expected.dropna().values
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)