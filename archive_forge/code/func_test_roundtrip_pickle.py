from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
@pytest.mark.parametrize('_offset', [CBMonthBegin, CBMonthEnd])
def test_roundtrip_pickle(self, _offset):

    def _check_roundtrip(obj):
        unpickled = tm.round_trip_pickle(obj)
        assert unpickled == obj
    _check_roundtrip(_offset())
    _check_roundtrip(_offset(2))
    _check_roundtrip(_offset() * 2)