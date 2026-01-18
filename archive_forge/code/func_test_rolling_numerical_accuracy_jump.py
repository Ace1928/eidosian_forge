from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_numerical_accuracy_jump():
    index = date_range(start='2020-01-01', end='2020-01-02', freq='60s').append(DatetimeIndex(['2020-01-03']))
    data = np.random.default_rng(2).random(len(index))
    df = DataFrame({'data': data}, index=index)
    result = df.rolling('60s').mean()
    tm.assert_frame_equal(result, df[['data']])