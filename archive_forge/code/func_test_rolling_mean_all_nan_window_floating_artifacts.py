from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('start, exp_values', [(1, [0.03, 0.0155, 0.0155, 0.011, 0.01025]), (2, [0.001, 0.001, 0.0015, 0.00366666])])
def test_rolling_mean_all_nan_window_floating_artifacts(start, exp_values):
    df = DataFrame([0.03, 0.03, 0.001, np.nan, 0.002, 0.008, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.005, 0.2])
    values = exp_values + [0.00366666, 0.005, 0.005, 0.008, np.nan, np.nan, 0.005, 0.1025]
    expected = DataFrame(values, index=list(range(start, len(values) + start)))
    result = df.iloc[start:].rolling(5, min_periods=0).mean()
    tm.assert_frame_equal(result, expected)