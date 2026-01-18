import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_values_interleave_non_unique_cols(self):
    df = DataFrame([[Timestamp('20130101'), 3.5], [Timestamp('20130102'), 4.5]], columns=['x', 'x'], index=[1, 2])
    df_unique = df.copy()
    df_unique.columns = ['x', 'y']
    assert df_unique.values.shape == df.values.shape
    tm.assert_numpy_array_equal(df_unique.values[0], df.values[0])
    tm.assert_numpy_array_equal(df_unique.values[1], df.values[1])