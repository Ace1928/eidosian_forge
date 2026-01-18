from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method', ['split', 'rsplit'])
@pytest.mark.parametrize('n', [None, 0])
def test_split_n(any_string_dtype, method, n):
    s = Series(['a b', pd.NA, 'b c'], dtype=any_string_dtype)
    expected = Series([['a', 'b'], pd.NA, ['b', 'c']])
    result = getattr(s.str, method)(' ', n=n)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)