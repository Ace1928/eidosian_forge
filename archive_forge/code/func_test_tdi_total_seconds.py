import numpy as np
import pytest
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_tdi_total_seconds(self):
    rng = timedelta_range('1 days, 10:11:12.100123456', periods=2, freq='s')
    expt = [1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1000000000.0, 1 * 86400 + 10 * 3600 + 11 * 60 + 13 + 100123456.0 / 1000000000.0]
    tm.assert_almost_equal(rng.total_seconds(), Index(expt))
    ser = Series(rng)
    s_expt = Series(expt, index=[0, 1])
    tm.assert_series_equal(ser.dt.total_seconds(), s_expt)
    ser[1] = np.nan
    s_expt = Series([1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1000000000.0, np.nan], index=[0, 1])
    tm.assert_series_equal(ser.dt.total_seconds(), s_expt)