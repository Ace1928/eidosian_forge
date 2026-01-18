import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_frame_dtype_fidelity():
    mi = MultiIndex.from_arrays([pd.date_range('19910905', periods=6, tz='US/Eastern'), [1, 1, 1, 2, 2, 2], pd.Categorical(['a', 'a', 'b', 'b', 'c', 'c'], ordered=True), ['x', 'x', 'y', 'z', 'x', 'y']], names=['dates', 'a', 'b', 'c'])
    original_dtypes = {name: mi.levels[i].dtype for i, name in enumerate(mi.names)}
    expected_df = DataFrame({'dates': pd.date_range('19910905', periods=6, tz='US/Eastern'), 'a': [1, 1, 1, 2, 2, 2], 'b': pd.Categorical(['a', 'a', 'b', 'b', 'c', 'c'], ordered=True), 'c': ['x', 'x', 'y', 'z', 'x', 'y']})
    df = mi.to_frame(index=False)
    df_dtypes = df.dtypes.to_dict()
    tm.assert_frame_equal(df, expected_df)
    assert original_dtypes == df_dtypes