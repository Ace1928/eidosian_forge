from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('partial_dtime', ['2019', '2019Q4', 'Dec 2019', '2019-12-31', '2019-12-31 23', '2019-12-31 23:59'])
def test_slice_end_of_period_resolution(self, partial_dtime):
    dti = date_range('2019-12-31 23:59:55.999999999', periods=10, freq='s')
    ser = Series(range(10), index=dti)
    result = ser[partial_dtime]
    expected = ser.iloc[:5]
    tm.assert_series_equal(result, expected)