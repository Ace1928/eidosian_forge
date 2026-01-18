from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('method', ['var', 'sum', 'mean', 'skew', 'kurt', 'min', 'max'])
def test_rolling_decreasing_indices(method):
    """
    Make sure that decreasing indices give the same results as increasing indices.

    GH 36933
    """
    df = DataFrame({'values': np.arange(-15, 10) ** 2})
    df_reverse = DataFrame({'values': df['values'][::-1]}, index=df.index[::-1])
    increasing = getattr(df.rolling(window=5), method)()
    decreasing = getattr(df_reverse.rolling(window=5), method)()
    assert np.abs(decreasing.values[::-1][:-4] - increasing.values[4:]).max() < 1e-12