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
def test_not_monotonic():
    rng = DatetimeIndex(['1/31/2000', '1/31/2001', '1/31/2002'])
    rng = rng[::-1]
    assert rng.inferred_freq == '-1YE-JAN'