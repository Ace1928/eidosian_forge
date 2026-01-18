import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_exclude_include_using_list_like(self):
    df = DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6, dtype='u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.date_range('now', periods=3).values})
    exclude = (np.datetime64,)
    include = (np.bool_, 'integer')
    r = df.select_dtypes(include=include, exclude=exclude)
    e = df[['b', 'c', 'e']]
    tm.assert_frame_equal(r, e)
    exclude = ('datetime',)
    include = ('bool', 'int64', 'int32')
    r = df.select_dtypes(include=include, exclude=exclude)
    e = df[['b', 'e']]
    tm.assert_frame_equal(r, e)