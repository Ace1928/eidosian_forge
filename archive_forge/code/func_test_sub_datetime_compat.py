from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_sub_datetime_compat(self, unit):
    ser = Series([datetime(2016, 8, 23, 12, tzinfo=pytz.utc), NaT]).dt.as_unit(unit)
    dt = datetime(2016, 8, 22, 12, tzinfo=pytz.utc)
    exp_unit = tm.get_finest_unit(unit, 'us')
    exp = Series([Timedelta('1 days'), NaT]).dt.as_unit(exp_unit)
    result = ser - dt
    tm.assert_series_equal(result, exp)
    result2 = ser - Timestamp(dt)
    tm.assert_series_equal(result2, exp)