from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('klass,expected', [(Timestamp, ['as_unit', 'astimezone', 'ceil', 'combine', 'ctime', 'date', 'day_name', 'dst', 'floor', 'fromisocalendar', 'fromisoformat', 'fromordinal', 'fromtimestamp', 'isocalendar', 'isoformat', 'isoweekday', 'month_name', 'now', 'replace', 'round', 'strftime', 'strptime', 'time', 'timestamp', 'timetuple', 'timetz', 'to_datetime64', 'to_numpy', 'to_pydatetime', 'today', 'toordinal', 'tz_convert', 'tz_localize', 'tzname', 'utcfromtimestamp', 'utcnow', 'utcoffset', 'utctimetuple', 'weekday']), (Timedelta, ['total_seconds'])])
def test_overlap_public_nat_methods(klass, expected):
    assert _get_overlap_public_nat_methods(klass) == expected