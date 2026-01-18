import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_putmask_td64(self):
    dti = date_range('2016-01-01', periods=9)
    tdi = dti - dti[0]
    idx = IntervalIndex.from_breaks(tdi)
    mask = np.zeros(idx.shape, dtype=bool)
    mask[0:3] = True
    result = idx.putmask(mask, idx[-1])
    expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
    tm.assert_index_equal(result, expected)