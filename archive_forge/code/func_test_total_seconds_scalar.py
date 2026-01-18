from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
def test_total_seconds_scalar(self):
    rng = Timedelta('1 days, 10:11:12.100123456')
    expt = 1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1000000000.0
    tm.assert_almost_equal(rng.total_seconds(), expt)
    rng = Timedelta(np.nan)
    assert np.isnan(rng.total_seconds())