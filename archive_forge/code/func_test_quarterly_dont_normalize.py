from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_quarterly_dont_normalize():
    date = datetime(2012, 3, 31, 5, 30)
    offsets = (BQuarterEnd, BQuarterBegin)
    for klass in offsets:
        result = date + klass()
        assert result.time() == date.time()