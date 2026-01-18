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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='None is cast to NaN')
def test_pivot_columns_is_none(self):
    df = DataFrame({None: [1], 'b': 2, 'c': 3})
    result = df.pivot(columns=None)
    expected = DataFrame({('b', 1): [2], ('c', 1): 3})
    tm.assert_frame_equal(result, expected)
    result = df.pivot(columns=None, index='b')
    expected = DataFrame({('c', 1): 3}, index=Index([2], name='b'))
    tm.assert_frame_equal(result, expected)
    result = df.pivot(columns=None, index='b', values='c')
    expected = DataFrame({1: 3}, index=Index([2], name='b'))
    tm.assert_frame_equal(result, expected)