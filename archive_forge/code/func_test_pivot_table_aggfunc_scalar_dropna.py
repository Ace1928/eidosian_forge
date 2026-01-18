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
def test_pivot_table_aggfunc_scalar_dropna(self, dropna):
    df = DataFrame({'A': ['one', 'two', 'one'], 'x': [3, np.nan, 2], 'y': [1, np.nan, np.nan]})
    result = pivot_table(df, columns='A', aggfunc='mean', dropna=dropna)
    data = [[2.5, np.nan], [1, np.nan]]
    col = Index(['one', 'two'], name='A')
    expected = DataFrame(data, index=['x', 'y'], columns=col)
    if dropna:
        expected = expected.dropna(axis='columns')
    tm.assert_frame_equal(result, expected)