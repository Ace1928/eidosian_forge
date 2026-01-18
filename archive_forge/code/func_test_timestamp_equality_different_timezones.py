from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_timestamp_equality_different_timezones(self):
    utc_range = date_range('1/1/2000', periods=20, tz='UTC')
    eastern_range = utc_range.tz_convert('US/Eastern')
    berlin_range = utc_range.tz_convert('Europe/Berlin')
    for a, b, c in zip(utc_range, eastern_range, berlin_range):
        assert a == b
        assert b == c
        assert a == c
    assert (utc_range == eastern_range).all()
    assert (utc_range == berlin_range).all()
    assert (berlin_range == eastern_range).all()