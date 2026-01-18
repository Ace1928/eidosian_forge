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
def test_sub_single_tz(self, unit):
    s1 = Series([Timestamp('2016-02-10', tz='America/Sao_Paulo')]).dt.as_unit(unit)
    s2 = Series([Timestamp('2016-02-08', tz='America/Sao_Paulo')]).dt.as_unit(unit)
    result = s1 - s2
    expected = Series([Timedelta('2days')]).dt.as_unit(unit)
    tm.assert_series_equal(result, expected)
    result = s2 - s1
    expected = Series([Timedelta('-2days')]).dt.as_unit(unit)
    tm.assert_series_equal(result, expected)