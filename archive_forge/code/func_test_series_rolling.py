import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from .utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('window', [5, 100])
@pytest.mark.parametrize('min_periods', [None, 5])
@pytest.mark.parametrize('method, kwargs', [('count', {}), ('sum', {}), ('mean', {}), ('var', {'ddof': 0}), ('std', {'ddof': 0}), ('min', {}), ('max', {}), ('skew', {}), ('kurt', {}), ('apply', {'func': np.sum}), ('rank', {}), ('sem', {'ddof': 0}), ('aggregate', {'func': np.sum}), ('agg', {'func': [np.sum, np.mean]}), ('quantile', {'q': 0.1}), ('median', {})])
def test_series_rolling(data, window, min_periods, method, kwargs):
    modin_series, pandas_series = create_test_series(data)
    if window > len(pandas_series):
        window = len(pandas_series)
    eval_general(modin_series, pandas_series, lambda series: getattr(series.rolling(window=window, min_periods=min_periods, win_type=None, center=True), method)(**kwargs))