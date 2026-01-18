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
@pytest.mark.parametrize('as_index', [True, False])
@pytest.mark.parametrize('values', [pd.to_datetime(['2020-01-01', '2020-02-01']), pd.to_timedelta([1, 2], unit='D'), PeriodIndex(['2020-01-01', '2020-02-01'], freq='D')])
@pytest.mark.parametrize('klass', [list, np.array, pd.array, pd.Series, pd.Index, pd.Categorical, pd.CategoricalIndex])
def test_searchsorted_datetimelike_with_listlike(values, klass, as_index):
    if not as_index:
        values = values._data
    result = values.searchsorted(klass(values))
    expected = np.array([0, 1], dtype=result.dtype)
    tm.assert_numpy_array_equal(result, expected)