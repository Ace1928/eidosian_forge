from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
def test_unary_nullable(self):
    df = pd.DataFrame({'a': pd.array([1, -2, 3, pd.NA], dtype='Int64'), 'b': pd.array([4.0, -5.0, 6.0, pd.NA], dtype='Float32'), 'c': pd.array([True, False, False, pd.NA], dtype='boolean'), 'd': np.array([True, False, False, True])})
    result = +df
    res_ufunc = np.positive(df)
    expected = df
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(res_ufunc, expected)
    result = -df
    res_ufunc = np.negative(df)
    expected = pd.DataFrame({'a': pd.array([-1, 2, -3, pd.NA], dtype='Int64'), 'b': pd.array([-4.0, 5.0, -6.0, pd.NA], dtype='Float32'), 'c': pd.array([False, True, True, pd.NA], dtype='boolean'), 'd': np.array([False, True, True, False])})
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(res_ufunc, expected)
    result = abs(df)
    res_ufunc = np.abs(df)
    expected = pd.DataFrame({'a': pd.array([1, 2, 3, pd.NA], dtype='Int64'), 'b': pd.array([4.0, 5.0, 6.0, pd.NA], dtype='Float32'), 'c': pd.array([True, False, False, pd.NA], dtype='boolean'), 'd': np.array([True, False, False, True])})
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(res_ufunc, expected)