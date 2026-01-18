from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('value', [pd.Timedelta(days=9), timedelta(days=9), np.timedelta64(9, 'D')])
def test_where_index_timedelta64(self, value):
    tdi = pd.timedelta_range('1 Day', periods=4)
    cond = np.array([True, False, False, True])
    expected = pd.TimedeltaIndex(['1 Day', value, value, '4 Days'])
    result = tdi.where(cond, value)
    tm.assert_index_equal(result, expected)
    dtnat = np.datetime64('NaT', 'ns')
    expected = pd.Index([tdi[0], dtnat, dtnat, tdi[3]], dtype=object)
    assert expected[1] is dtnat
    result = tdi.where(cond, dtnat)
    tm.assert_index_equal(result, expected)