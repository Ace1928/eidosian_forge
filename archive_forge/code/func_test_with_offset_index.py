from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
@pytest.mark.parametrize('td', [Timedelta(hours=2), Timedelta(hours=2).to_pytimedelta(), Timedelta(hours=2).to_timedelta64()], ids=lambda x: type(x))
def test_with_offset_index(self, td, dt, offset):
    dti = DatetimeIndex([dt])
    expected = DatetimeIndex([datetime(2008, 1, 2, 2)])
    result = dti + (td + offset)
    tm.assert_index_equal(result, expected)
    result = dti + (offset + td)
    tm.assert_index_equal(result, expected)