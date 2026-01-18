import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('periods', [1, 2, 3, 4])
def test_shift_preserve_freqstr(self, periods, frame_or_series):
    obj = frame_or_series(range(periods), index=date_range('2016-1-1 00:00:00', periods=periods, freq='h'))
    result = obj.shift(1, '2h')
    expected = frame_or_series(range(periods), index=date_range('2016-1-1 02:00:00', periods=periods, freq='h'))
    tm.assert_equal(result, expected)