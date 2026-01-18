import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_not_an_attr_but_still_valid_dtype(self, using_infer_string):
    df = DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6).astype('u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.date_range('now', periods=3).values})
    df['g'] = df.f.diff()
    assert not hasattr(np, 'u8')
    r = df.select_dtypes(include=['i8', 'O'], exclude=['timedelta'])
    if using_infer_string:
        e = df[['b']]
    else:
        e = df[['a', 'b']]
    tm.assert_frame_equal(r, e)
    r = df.select_dtypes(include=['i8', 'O', 'timedelta64[ns]'])
    if using_infer_string:
        e = df[['b', 'g']]
    else:
        e = df[['a', 'b', 'g']]
    tm.assert_frame_equal(r, e)