from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_quantile_np_percentile():
    row = 10
    col = 5
    idx = date_range('20100101', periods=row, freq='B')
    df = DataFrame(np.random.default_rng(2).random(row * col).reshape((row, -1)), index=idx)
    df_quantile = df.quantile([0.25, 0.5, 0.75], axis=0)
    np_percentile = np.percentile(df, [25, 50, 75], axis=0)
    tm.assert_almost_equal(df_quantile.values, np.array(np_percentile))