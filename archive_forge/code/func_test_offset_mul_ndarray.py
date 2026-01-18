from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def test_offset_mul_ndarray(self, offset_types):
    off = _create_offset(offset_types)
    expected = np.array([[off, off * 2], [off * 3, off * 4]])
    result = np.array([[1, 2], [3, 4]]) * off
    tm.assert_numpy_array_equal(result, expected)
    result = off * np.array([[1, 2], [3, 4]])
    tm.assert_numpy_array_equal(result, expected)