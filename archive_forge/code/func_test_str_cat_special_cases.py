import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
def test_str_cat_special_cases():
    s = Series(['a', 'b', 'c', 'd'])
    t = Series(['d', 'a', 'e', 'b'], index=[3, 0, 4, 1])
    expected = Series(['aaa', 'bbb', 'c-c', 'ddd', '-e-'])
    result = s.str.cat(iter([t, s.values]), join='outer', na_rep='-')
    tm.assert_series_equal(result, expected)
    expected = Series(['aa-', 'd-d'], index=[0, 3])
    result = s.str.cat([t.loc[[0]], t.loc[[3]]], join='right', na_rep='-')
    tm.assert_series_equal(result, expected)