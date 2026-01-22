from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
class TestCommonCBM:

    @pytest.mark.parametrize('offset2', [CBMonthBegin(2), CBMonthEnd(2)])
    def test_eq(self, offset2):
        assert offset2 == offset2

    @pytest.mark.parametrize('offset2', [CBMonthBegin(2), CBMonthEnd(2)])
    def test_hash(self, offset2):
        assert hash(offset2) == hash(offset2)

    @pytest.mark.parametrize('_offset', [CBMonthBegin, CBMonthEnd])
    def test_roundtrip_pickle(self, _offset):

        def _check_roundtrip(obj):
            unpickled = tm.round_trip_pickle(obj)
            assert unpickled == obj
        _check_roundtrip(_offset())
        _check_roundtrip(_offset(2))
        _check_roundtrip(_offset() * 2)

    @pytest.mark.parametrize('_offset', [CBMonthBegin, CBMonthEnd])
    def test_copy(self, _offset):
        off = _offset(weekmask='Mon Wed Fri')
        assert off == off.copy()