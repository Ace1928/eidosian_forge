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
def test_pivot_periods(self, method):
    df = DataFrame({'p1': [pd.Period('2013-01-01', 'D'), pd.Period('2013-01-02', 'D'), pd.Period('2013-01-01', 'D'), pd.Period('2013-01-02', 'D')], 'p2': [pd.Period('2013-01', 'M'), pd.Period('2013-01', 'M'), pd.Period('2013-02', 'M'), pd.Period('2013-02', 'M')], 'data1': np.arange(4, dtype='int64'), 'data2': np.arange(4, dtype='int64')})
    exp_col1 = Index(['data1', 'data1', 'data2', 'data2'])
    exp_col2 = pd.PeriodIndex(['2013-01', '2013-02'] * 2, name='p2', freq='M')
    exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
    expected = DataFrame([[0, 2, 0, 2], [1, 3, 1, 3]], index=pd.PeriodIndex(['2013-01-01', '2013-01-02'], name='p1', freq='D'), columns=exp_col)
    if method:
        pv = df.pivot(index='p1', columns='p2')
    else:
        pv = pd.pivot(df, index='p1', columns='p2')
    tm.assert_frame_equal(pv, expected)
    expected = DataFrame([[0, 2], [1, 3]], index=pd.PeriodIndex(['2013-01-01', '2013-01-02'], name='p1', freq='D'), columns=pd.PeriodIndex(['2013-01', '2013-02'], name='p2', freq='M'))
    if method:
        pv = df.pivot(index='p1', columns='p2', values='data1')
    else:
        pv = pd.pivot(df, index='p1', columns='p2', values='data1')
    tm.assert_frame_equal(pv, expected)