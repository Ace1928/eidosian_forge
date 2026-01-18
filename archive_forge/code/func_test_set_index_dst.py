from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_index_dst(self):
    di = date_range('2006-10-29 00:00:00', periods=3, freq='h', tz='US/Pacific')
    df = DataFrame(data={'a': [0, 1, 2], 'b': [3, 4, 5]}, index=di).reset_index()
    res = df.set_index('index')
    exp = DataFrame(data={'a': [0, 1, 2], 'b': [3, 4, 5]}, index=Index(di, name='index'))
    exp.index = exp.index._with_freq(None)
    tm.assert_frame_equal(res, exp)
    res = df.set_index(['index', 'a'])
    exp_index = MultiIndex.from_arrays([di, [0, 1, 2]], names=['index', 'a'])
    exp = DataFrame({'b': [3, 4, 5]}, index=exp_index)
    tm.assert_frame_equal(res, exp)