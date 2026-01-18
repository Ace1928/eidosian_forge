import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from .utils import (
@pytest.mark.parametrize('closed', ['both', 'right'])
def test_series_dt_index(closed):
    index = pandas.date_range('1/1/2000', periods=12, freq='min')
    pandas_series = pandas.Series(range(12), index=index)
    modin_series = pd.Series(range(12), index=index)
    pandas_rolled = pandas_series.rolling('3s', closed=closed)
    modin_rolled = modin_series.rolling('3s', closed=closed)
    df_equals(modin_rolled.count(), pandas_rolled.count())
    df_equals(modin_rolled.skew(), pandas_rolled.skew())
    df_equals(modin_rolled.apply(np.sum, raw=True), pandas_rolled.apply(np.sum, raw=True))
    df_equals(modin_rolled.aggregate(np.sum), pandas_rolled.aggregate(np.sum))
    df_equals(modin_rolled.quantile(0.1), pandas_rolled.quantile(0.1))