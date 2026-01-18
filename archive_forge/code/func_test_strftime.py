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
def test_strftime(self, arr1d):
    arr = arr1d
    result = arr.strftime('%Y')
    expected = np.array([per.strftime('%Y') for per in arr], dtype=object)
    tm.assert_numpy_array_equal(result, expected)