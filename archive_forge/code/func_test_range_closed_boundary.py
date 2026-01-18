from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_range_closed_boundary(self, inclusive_endpoints_fixture):
    right_boundary = date_range('2015-09-12', '2015-12-01', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
    left_boundary = date_range('2015-09-01', '2015-09-12', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
    both_boundary = date_range('2015-09-01', '2015-12-01', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
    neither_boundary = date_range('2015-09-11', '2015-09-12', freq='QS-MAR', inclusive=inclusive_endpoints_fixture)
    expected_right = both_boundary
    expected_left = both_boundary
    expected_both = both_boundary
    if inclusive_endpoints_fixture == 'right':
        expected_left = both_boundary[1:]
    elif inclusive_endpoints_fixture == 'left':
        expected_right = both_boundary[:-1]
    elif inclusive_endpoints_fixture == 'both':
        expected_right = both_boundary[1:]
        expected_left = both_boundary[:-1]
    expected_neither = both_boundary[1:-1]
    tm.assert_index_equal(right_boundary, expected_right)
    tm.assert_index_equal(left_boundary, expected_left)
    tm.assert_index_equal(both_boundary, expected_both)
    tm.assert_index_equal(neither_boundary, expected_neither)