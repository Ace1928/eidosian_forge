from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
@pytest.mark.parametrize('offset', [BQuarterBegin(), BQuarterEnd()])
def test_on_offset(offset):
    dates = [datetime(2016, m, d) for m in [10, 11, 12] for d in [1, 2, 3, 28, 29, 30, 31] if not (m == 11 and d == 31)]
    for date in dates:
        res = offset.is_on_offset(date)
        slow_version = date == date + offset - offset
        assert res == slow_version