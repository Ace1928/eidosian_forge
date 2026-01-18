from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
@pytest.mark.parametrize('method', [True, False])
def test_pivot_index_with_nan_dates(self, method):
    df = DataFrame({'a': date_range('2014-02-01', periods=6, freq='D'), 'c': 100 + np.arange(6)})
    df['b'] = df['a'] - pd.Timestamp('2014-02-02')
    df.loc[1, 'a'] = df.loc[3, 'a'] = np.nan
    df.loc[1, 'b'] = df.loc[4, 'b'] = np.nan
    if method:
        pv = df.pivot(index='a', columns='b', values='c')
    else:
        pv = pd.pivot(df, index='a', columns='b', values='c')
    assert pv.notna().values.sum() == len(df)
    for _, row in df.iterrows():
        assert pv.loc[row['a'], row['b']] == row['c']
    if method:
        result = df.pivot(index='b', columns='a', values='c')
    else:
        result = pd.pivot(df, index='b', columns='a', values='c')
    tm.assert_frame_equal(result, pv.T)