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
@pytest.mark.parametrize('freq', ['ME', 'ms', 's'])
def test_series_datetime_index(freq):
    s = Series(date_range('20130101', periods=10, freq=freq))
    inferred = frequencies.infer_freq(s)
    assert inferred == freq