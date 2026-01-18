from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_offset_corner_case(self):
    offset = BQuarterEnd(n=-1, startingMonth=1)
    assert datetime(2010, 1, 31) + offset == datetime(2010, 1, 29)