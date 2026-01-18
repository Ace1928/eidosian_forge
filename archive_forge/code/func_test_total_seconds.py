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
def test_total_seconds(self, timedelta_index):
    tdi = timedelta_index
    arr = tdi._data
    expected = tdi.total_seconds()
    result = arr.total_seconds()
    tm.assert_numpy_array_equal(result, expected.values)