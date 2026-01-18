from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
def test_tz_convert_roundtrip(self, tz_aware_fixture):
    tz = tz_aware_fixture
    idx1 = date_range(start='2014-01-01', end='2014-12-31', freq='ME', tz='UTC')
    exp1 = date_range(start='2014-01-01', end='2014-12-31', freq='ME')
    idx2 = date_range(start='2014-01-01', end='2014-12-31', freq='D', tz='UTC')
    exp2 = date_range(start='2014-01-01', end='2014-12-31', freq='D')
    idx3 = date_range(start='2014-01-01', end='2014-03-01', freq='h', tz='UTC')
    exp3 = date_range(start='2014-01-01', end='2014-03-01', freq='h')
    idx4 = date_range(start='2014-08-01', end='2014-10-31', freq='min', tz='UTC')
    exp4 = date_range(start='2014-08-01', end='2014-10-31', freq='min')
    for idx, expected in [(idx1, exp1), (idx2, exp2), (idx3, exp3), (idx4, exp4)]:
        converted = idx.tz_convert(tz)
        reset = converted.tz_convert(None)
        tm.assert_index_equal(reset, expected)
        assert reset.tzinfo is None
        expected = converted.tz_convert('UTC').tz_localize(None)
        expected = expected._with_freq('infer')
        tm.assert_index_equal(reset, expected)