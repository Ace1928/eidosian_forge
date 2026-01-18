import numpy as np
from pandas.core.dtypes.common import is_float_dtype
from pandas import (
import pandas._testing as tm
def test_set_value_with_index_dtype_change(self):
    df_orig = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=range(3), columns=list('ABC'))
    df = df_orig.copy()
    df._set_value('C', 2, 1.0)
    assert list(df.index) == list(df_orig.index) + ['C']
    df = df_orig.copy()
    df.loc['C', 2] = 1.0
    assert list(df.index) == list(df_orig.index) + ['C']
    df = df_orig.copy()
    df._set_value('C', 'D', 1.0)
    assert list(df.index) == list(df_orig.index) + ['C']
    assert list(df.columns) == list(df_orig.columns) + ['D']
    df = df_orig.copy()
    df.loc['C', 'D'] = 1.0
    assert list(df.index) == list(df_orig.index) + ['C']
    assert list(df.columns) == list(df_orig.columns) + ['D']