import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_interpolation(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]}, index=[1, 2, 3])
    result = df.quantile(0.5, axis=1, interpolation='nearest')
    expected = Series([1, 2, 3], index=[1, 2, 3], name=0.5)
    tm.assert_series_equal(result, expected)
    exp = np.percentile(np.array([[1, 2, 3], [2, 3, 4]]), 0.5, axis=0, method='nearest')
    expected = Series(exp, index=[1, 2, 3], name=0.5, dtype='int64')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [1.0, 2.0, 3.0], 'B': [2.0, 3.0, 4.0]}, index=[1, 2, 3])
    result = df.quantile(0.5, axis=1, interpolation='nearest')
    expected = Series([1.0, 2.0, 3.0], index=[1, 2, 3], name=0.5)
    tm.assert_series_equal(result, expected)
    exp = np.percentile(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]), 0.5, axis=0, method='nearest')
    expected = Series(exp, index=[1, 2, 3], name=0.5, dtype='float64')
    tm.assert_series_equal(result, expected)
    result = df.quantile([0.5, 0.75], axis=1, interpolation='lower')
    expected = DataFrame({1: [1.0, 1.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}, index=[0.5, 0.75])
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'x': [], 'y': []})
    q = df.quantile(0.1, axis=0, interpolation='higher')
    assert np.isnan(q['x']) and np.isnan(q['y'])
    df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=['a', 'b', 'c'])
    result = df.quantile([0.25, 0.5], interpolation='midpoint')
    expected = DataFrame([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]], index=[0.25, 0.5], columns=['a', 'b', 'c'])
    tm.assert_frame_equal(result, expected)