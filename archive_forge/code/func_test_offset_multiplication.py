from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
@pytest.mark.parametrize('n_months, scaling_factor, start_timestamp, expected_timestamp', [(1, 2, '2020-01-30', '2020-03-30'), (2, 1, '2020-01-30', '2020-03-30'), (1, 0, '2020-01-30', '2020-01-30'), (2, 0, '2020-01-30', '2020-01-30'), (1, -1, '2020-01-30', '2019-12-30'), (2, -1, '2020-01-30', '2019-11-30')])
def test_offset_multiplication(n_months, scaling_factor, start_timestamp, expected_timestamp):
    mo1 = DateOffset(months=n_months)
    startscalar = Timestamp(start_timestamp)
    startarray = Series([startscalar])
    resultscalar = startscalar + mo1 * scaling_factor
    resultarray = startarray + mo1 * scaling_factor
    expectedscalar = Timestamp(expected_timestamp)
    expectedarray = Series([expectedscalar])
    assert resultscalar == expectedscalar
    tm.assert_series_equal(resultarray, expectedarray)