from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_to_timestamp_out_of_bounds(self):
    pi = pd.period_range('1500', freq='Y', periods=3)
    msg = 'Out of bounds nanosecond timestamp: 1500-01-01 00:00:00'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        pi.to_timestamp()
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        pi._data.to_timestamp()