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
def _get_expected_range(begin_to_match, end_to_match, both_range, inclusive_endpoints):
    """Helper to get expected range from a both inclusive range"""
    left_match = begin_to_match == both_range[0]
    right_match = end_to_match == both_range[-1]
    if inclusive_endpoints == 'left' and right_match:
        expected_range = both_range[:-1]
    elif inclusive_endpoints == 'right' and left_match:
        expected_range = both_range[1:]
    elif inclusive_endpoints == 'neither' and left_match and right_match:
        expected_range = both_range[1:-1]
    elif inclusive_endpoints == 'neither' and right_match:
        expected_range = both_range[:-1]
    elif inclusive_endpoints == 'neither' and left_match:
        expected_range = both_range[1:]
    elif inclusive_endpoints == 'both':
        expected_range = both_range[:]
    else:
        expected_range = both_range[:]
    return expected_range