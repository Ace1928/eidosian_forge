from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_range_casting():
    idx = np.arange(120).astype(np.int64)
    dp = DeterministicProcess(idx, constant=True, order=1, seasonal=True, period=12)
    idx = pd.RangeIndex(0, 120)
    dp2 = DeterministicProcess(idx, constant=True, order=1, seasonal=True, period=12)
    pd.testing.assert_frame_equal(dp.in_sample(), dp2.in_sample())
    pd.testing.assert_frame_equal(dp.range(100, 150), dp2.range(100, 150))