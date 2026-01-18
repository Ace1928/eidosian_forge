from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_rsub_dtscalars(self, tz_naive_fixture):
    td = Timedelta(1235345642000)
    ts = Timestamp('2021-01-01', tz=tz_naive_fixture)
    other = ts + td
    assert other - ts == td
    assert other.to_pydatetime() - ts == td
    if tz_naive_fixture is None:
        assert other.to_datetime64() - ts == td
    else:
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            other.to_datetime64() - ts