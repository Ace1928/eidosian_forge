from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_ufunc_fallback(data):
    a = data[:5]
    s = pd.Series(a, index=range(3, 8))
    result = np.abs(s)
    expected = pd.Series(np.abs(a), index=range(3, 8))
    tm.assert_series_equal(result, expected)