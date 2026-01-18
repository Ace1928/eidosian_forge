from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_more_regex_split(any_string_dtype):
    values = Series(['a,b_c', 'c_d,e', np.nan, 'f,g,h'], dtype=any_string_dtype)
    result = values.str.split('[,_]')
    exp = Series([['a', 'b', 'c'], ['c', 'd', 'e'], np.nan, ['f', 'g', 'h']])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)