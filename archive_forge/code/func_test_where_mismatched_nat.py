from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_where_mismatched_nat(self, tz_aware_fixture):
    tz = tz_aware_fixture
    dti = date_range('2013-01-01', periods=3, tz=tz)
    cond = np.array([True, False, True])
    tdnat = np.timedelta64('NaT', 'ns')
    expected = Index([dti[0], tdnat, dti[2]], dtype=object)
    assert expected[1] is tdnat
    result = dti.where(cond, tdnat)
    tm.assert_index_equal(result, expected)