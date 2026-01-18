from datetime import timedelta
import pytest
import pytz
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.errors import PerformanceWarning
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.util.version import Version
def test_springforward_plural(self):
    for tz, utc_offsets in self.timezone_utc_offsets.items():
        hrs_pre = utc_offsets['utc_offset_standard']
        hrs_post = utc_offsets['utc_offset_daylight']
        self._test_all_offsets(n=3, tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz), expected_utc_offset=hrs_post)