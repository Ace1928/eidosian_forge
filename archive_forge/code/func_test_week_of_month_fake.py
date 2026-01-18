from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.offsets import _get_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import is_platform_windows
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.tools.datetimes import to_datetime
from pandas.tseries import (
def test_week_of_month_fake():
    index = DatetimeIndex(['2013-08-27', '2013-10-01', '2013-10-29', '2013-11-26'])
    assert frequencies.infer_freq(index) != 'WOM-4TUE'