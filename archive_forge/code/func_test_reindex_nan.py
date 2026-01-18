from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_nan(self):
    df = DataFrame([[1, 2], [3, 5], [7, 11], [9, 23]], index=[2, np.nan, 1, 5], columns=['joe', 'jim'])
    i, j = ([np.nan, 5, 5, np.nan, 1, 2, np.nan], [1, 3, 3, 1, 2, 0, 1])
    tm.assert_frame_equal(df.reindex(i), df.iloc[j])
    df.index = df.index.astype('object')
    tm.assert_frame_equal(df.reindex(i), df.iloc[j], check_index_type=False)
    df = DataFrame({'other': ['a', 'b', np.nan, 'c'], 'date': ['2015-03-22', np.nan, '2012-01-08', np.nan], 'amount': [2, 3, 4, 5]})
    df['date'] = pd.to_datetime(df.date)
    df['delta'] = (pd.to_datetime('2015-06-18') - df['date']).shift(1)
    left = df.set_index(['delta', 'other', 'date']).reset_index()
    right = df.reindex(columns=['delta', 'other', 'date', 'amount'])
    tm.assert_frame_equal(left, right)