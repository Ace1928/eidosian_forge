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
def test_pivot_table_not_series(self):
    df = DataFrame({'col1': [3, 4, 5], 'col2': ['C', 'D', 'E'], 'col3': [1, 3, 9]})
    result = df.pivot_table('col1', index=['col3', 'col2'], aggfunc='sum')
    m = MultiIndex.from_arrays([[1, 3, 9], ['C', 'D', 'E']], names=['col3', 'col2'])
    expected = DataFrame([3, 4, 5], index=m, columns=['col1'])
    tm.assert_frame_equal(result, expected)
    result = df.pivot_table('col1', index='col3', columns='col2', aggfunc='sum')
    expected = DataFrame([[3, np.nan, np.nan], [np.nan, 4, np.nan], [np.nan, np.nan, 5]], index=Index([1, 3, 9], name='col3'), columns=Index(['C', 'D', 'E'], name='col2'))
    tm.assert_frame_equal(result, expected)
    result = df.pivot_table('col1', index='col3', aggfunc=['sum'])
    m = MultiIndex.from_arrays([['sum'], ['col1']])
    expected = DataFrame([3, 4, 5], index=Index([1, 3, 9], name='col3'), columns=m)
    tm.assert_frame_equal(result, expected)