from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_cache_updating_slices(self, using_copy_on_write, warn_copy_on_write):
    expected = DataFrame({'A': [600, 600, 600]}, index=date_range('5/7/2014', '5/9/2014'))
    out = DataFrame({'A': [0, 0, 0]}, index=date_range('5/7/2014', '5/9/2014'))
    df = DataFrame({'C': ['A', 'A', 'A'], 'D': [100, 200, 300]})
    six = Timestamp('5/7/2014')
    eix = Timestamp('5/9/2014')
    for ix, row in df.iterrows():
        out.loc[six:eix, row['C']] = out.loc[six:eix, row['C']] + row['D']
    tm.assert_frame_equal(out, expected)
    tm.assert_series_equal(out['A'], expected['A'])
    out = DataFrame({'A': [0, 0, 0]}, index=date_range('5/7/2014', '5/9/2014'))
    out_original = out.copy()
    for ix, row in df.iterrows():
        v = out[row['C']][six:eix] + row['D']
        with tm.raises_chained_assignment_error(ix == 0 or warn_copy_on_write or using_copy_on_write):
            out[row['C']][six:eix] = v
    if not using_copy_on_write:
        tm.assert_frame_equal(out, expected)
        tm.assert_series_equal(out['A'], expected['A'])
    else:
        tm.assert_frame_equal(out, out_original)
        tm.assert_series_equal(out['A'], out_original['A'])
    out = DataFrame({'A': [0, 0, 0]}, index=date_range('5/7/2014', '5/9/2014'))
    for ix, row in df.iterrows():
        out.loc[six:eix, row['C']] += row['D']
    tm.assert_frame_equal(out, expected)
    tm.assert_series_equal(out['A'], expected['A'])