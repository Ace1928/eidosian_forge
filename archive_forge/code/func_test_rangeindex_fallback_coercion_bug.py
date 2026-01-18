import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_rangeindex_fallback_coercion_bug():
    df1 = pd.DataFrame(np.arange(100).reshape((10, 10)))
    df2 = pd.DataFrame(np.arange(100).reshape((10, 10)))
    df = pd.concat({'df1': df1.stack(future_stack=True), 'df2': df2.stack(future_stack=True)}, axis=1)
    df.index.names = ['fizz', 'buzz']
    expected = pd.DataFrame({'df2': np.arange(100), 'df1': np.arange(100)}, index=MultiIndex.from_product([range(10), range(10)], names=['fizz', 'buzz']))
    tm.assert_frame_equal(df, expected, check_like=True)
    result = df.index.get_level_values('fizz')
    expected = Index(np.arange(10, dtype=np.int64), name='fizz').repeat(10)
    tm.assert_index_equal(result, expected)
    result = df.index.get_level_values('buzz')
    expected = Index(np.tile(np.arange(10, dtype=np.int64), 10), name='buzz')
    tm.assert_index_equal(result, expected)