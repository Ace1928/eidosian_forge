import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_ohlc_ea_dtypes(any_numeric_ea_dtype):
    df = DataFrame({'a': [1, 1, 2, 3, 4, 4], 'b': [22, 11, pd.NA, 10, 20, pd.NA]}, dtype=any_numeric_ea_dtype)
    gb = df.groupby('a')
    result = gb.ohlc()
    expected = DataFrame([[22, 22, 11, 11], [pd.NA] * 4, [10] * 4, [20] * 4], columns=MultiIndex.from_product([['b'], ['open', 'high', 'low', 'close']]), index=Index([1, 2, 3, 4], dtype=any_numeric_ea_dtype, name='a'), dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)
    gb2 = df.groupby('a', as_index=False)
    result2 = gb2.ohlc()
    expected2 = expected.reset_index()
    tm.assert_frame_equal(result2, expected2)