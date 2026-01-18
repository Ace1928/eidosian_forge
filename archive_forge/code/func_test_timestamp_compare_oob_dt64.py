from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
def test_timestamp_compare_oob_dt64(self):
    us = np.timedelta64(1, 'us')
    other = np.datetime64(Timestamp.min).astype('M8[us]')
    assert Timestamp.min > other
    other = np.datetime64(Timestamp.max).astype('M8[us]')
    assert Timestamp.max > other
    assert other < Timestamp.max
    assert Timestamp.max < other + us
    other = datetime(9999, 9, 9)
    assert Timestamp.min < other
    assert other > Timestamp.min
    assert Timestamp.max < other
    assert other > Timestamp.max
    other = datetime(1, 1, 1)
    assert Timestamp.max > other
    assert other < Timestamp.max
    assert Timestamp.min > other
    assert other < Timestamp.min