from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.parametrize('dtype, na_value', [('float64', np.nan), ('Float64', np.nan), ('Float64', pd.NA), ('Int64', pd.NA)])
@pytest.mark.parametrize('test_multiindex', [True, False])
def test_stack_preserves_na(dtype, na_value, test_multiindex):
    if test_multiindex:
        index = MultiIndex.from_arrays(2 * [Index([na_value], dtype=dtype)])
    else:
        index = Index([na_value], dtype=dtype)
    df = DataFrame({'a': [1]}, index=index)
    result = df.stack(future_stack=True)
    if test_multiindex:
        expected_index = MultiIndex.from_arrays([Index([na_value], dtype=dtype), Index([na_value], dtype=dtype), Index(['a'])])
    else:
        expected_index = MultiIndex.from_arrays([Index([na_value], dtype=dtype), Index(['a'])])
    expected = Series(1, index=expected_index)
    tm.assert_series_equal(result, expected)