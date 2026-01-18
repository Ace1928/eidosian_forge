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
def test_pivot_dtypes(self):
    f = DataFrame({'a': ['cat', 'bat', 'cat', 'bat'], 'v': [1, 2, 3, 4], 'i': ['a', 'b', 'a', 'b']})
    assert f.dtypes['v'] == 'int64'
    z = pivot_table(f, values='v', index=['a'], columns=['i'], fill_value=0, aggfunc='sum')
    result = z.dtypes
    expected = Series([np.dtype('int64')] * 2, index=Index(list('ab'), name='i'))
    tm.assert_series_equal(result, expected)
    f = DataFrame({'a': ['cat', 'bat', 'cat', 'bat'], 'v': [1.5, 2.5, 3.5, 4.5], 'i': ['a', 'b', 'a', 'b']})
    assert f.dtypes['v'] == 'float64'
    z = pivot_table(f, values='v', index=['a'], columns=['i'], fill_value=0, aggfunc='mean')
    result = z.dtypes
    expected = Series([np.dtype('float64')] * 2, index=Index(list('ab'), name='i'))
    tm.assert_series_equal(result, expected)