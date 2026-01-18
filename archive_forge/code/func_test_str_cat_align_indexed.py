import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.mark.parametrize('join', ['left', 'outer', 'inner', 'right'])
def test_str_cat_align_indexed(index_or_series, join):
    box = index_or_series
    s = Series(['a', 'b', 'c', 'd'], index=['a', 'b', 'c', 'd'])
    t = Series(['D', 'A', 'E', 'B'], index=['d', 'a', 'e', 'b'])
    sa, ta = s.align(t, join=join)
    expected = sa.str.cat(ta, na_rep='-')
    if box == Index:
        s = Index(s)
        sa = Index(sa)
        expected = Index(expected)
    result = s.str.cat(t, join=join, na_rep='-')
    tm.assert_equal(result, expected)