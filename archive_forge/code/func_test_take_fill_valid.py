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
def test_take_fill_valid(self, arr1d):
    arr = arr1d
    value = NaT._value
    msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
    with pytest.raises(TypeError, match=msg):
        arr.take([-1, 1], allow_fill=True, fill_value=value)
    value = np.timedelta64('NaT', 'ns')
    with pytest.raises(TypeError, match=msg):
        arr.take([-1, 1], allow_fill=True, fill_value=value)