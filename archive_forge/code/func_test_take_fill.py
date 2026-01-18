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
def test_take_fill(self, arr1d):
    arr = arr1d
    result = arr.take([-1, 1], allow_fill=True, fill_value=None)
    assert result[0] is NaT
    result = arr.take([-1, 1], allow_fill=True, fill_value=np.nan)
    assert result[0] is NaT
    result = arr.take([-1, 1], allow_fill=True, fill_value=NaT)
    assert result[0] is NaT