from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def _check_offsetfunc_works(self, offset, funcname, dt, expected, normalize=False):
    if normalize and issubclass(offset, Tick):
        return
    offset_s = _create_offset(offset, normalize=normalize)
    func = getattr(offset_s, funcname)
    result = func(dt)
    assert isinstance(result, Timestamp)
    assert result == expected
    result = func(Timestamp(dt))
    assert isinstance(result, Timestamp)
    assert result == expected
    ts = Timestamp(dt) + Nano(5)
    with tm.assert_produces_warning(None):
        result = func(ts)
    assert isinstance(result, Timestamp)
    if normalize is False:
        assert result == expected + Nano(5)
    else:
        assert result == expected
    if isinstance(dt, np.datetime64):
        return
    for tz in [None, 'UTC', 'Asia/Tokyo', 'US/Eastern', 'dateutil/Asia/Tokyo', 'dateutil/US/Pacific']:
        expected_localize = expected.tz_localize(tz)
        tz_obj = timezones.maybe_get_tz(tz)
        dt_tz = conversion.localize_pydatetime(dt, tz_obj)
        result = func(dt_tz)
        assert isinstance(result, Timestamp)
        assert result == expected_localize
        result = func(Timestamp(dt, tz=tz))
        assert isinstance(result, Timestamp)
        assert result == expected_localize
        ts = Timestamp(dt, tz=tz) + Nano(5)
        with tm.assert_produces_warning(None):
            result = func(ts)
        assert isinstance(result, Timestamp)
        if normalize is False:
            assert result == expected_localize + Nano(5)
        else:
            assert result == expected_localize