import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('f', [lambda x: x.expanding().cov(pairwise=True), lambda x: x.expanding().corr(pairwise=True), lambda x: x.rolling(window=3).cov(pairwise=True), lambda x: x.rolling(window=3).corr(pairwise=True), lambda x: x.ewm(com=3).cov(pairwise=True), lambda x: x.ewm(com=3).corr(pairwise=True)])
def test_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):
    result = f(pairwise_frames)
    tm.assert_index_equal(result.index.levels[0], pairwise_frames.index, check_names=False)
    tm.assert_index_equal(safe_sort(result.index.levels[1]), safe_sort(pairwise_frames.columns.unique()))
    tm.assert_index_equal(result.columns, pairwise_frames.columns)
    expected = f(pairwise_target_frame)
    result = result.dropna().values
    expected = expected.dropna().values
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)