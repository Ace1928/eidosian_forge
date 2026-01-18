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
@pytest.mark.parametrize('values', [pd.to_datetime(['2020-01-01', '2020-02-01']), pd.to_timedelta([1, 2], unit='D'), PeriodIndex(['2020-01-01', '2020-02-01'], freq='D')])
@pytest.mark.parametrize('arg', [[1, 2], ['a', 'b'], [Timestamp('2020-01-01', tz='Europe/London')] * 2])
def test_searchsorted_datetimelike_with_listlike_invalid_dtype(values, arg):
    msg = '[Unexpected type|Cannot compare]'
    with pytest.raises(TypeError, match=msg):
        values.searchsorted(arg)