from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_setitem_clears_freq(self):
    a = pd.date_range('2000', periods=2, freq='D', tz='US/Central')._data
    a[0] = pd.Timestamp('2000', tz='US/Central')
    assert a.freq is None