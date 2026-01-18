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
def test_pivot_table_multiindex_columns_doctest_case(self):
    df = DataFrame({'A': ['foo', 'foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar'], 'B': ['one', 'one', 'one', 'two', 'two', 'one', 'one', 'two', 'two'], 'C': ['small', 'large', 'large', 'small', 'small', 'large', 'small', 'small', 'large'], 'D': [1, 2, 2, 3, 3, 4, 5, 6, 7], 'E': [2, 4, 5, 5, 6, 6, 8, 9, 9]})
    table = pivot_table(df, values=['D', 'E'], index=['A', 'C'], aggfunc={'D': 'mean', 'E': ['min', 'max', 'mean']})
    cols = MultiIndex.from_tuples([('D', 'mean'), ('E', 'max'), ('E', 'mean'), ('E', 'min')])
    index = MultiIndex.from_tuples([('bar', 'large'), ('bar', 'small'), ('foo', 'large'), ('foo', 'small')], names=['A', 'C'])
    vals = np.array([[5.5, 9.0, 7.5, 6.0], [5.5, 9.0, 8.5, 8.0], [2.0, 5.0, 4.5, 4.0], [2.33333333, 6.0, 4.33333333, 2.0]])
    expected = DataFrame(vals, columns=cols, index=index)
    expected['E', 'min'] = expected['E', 'min'].astype(np.int64)
    expected['E', 'max'] = expected['E', 'max'].astype(np.int64)
    tm.assert_frame_equal(table, expected)