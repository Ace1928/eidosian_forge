from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
def test_numpy_reduction(test_series):
    result = test_series.resample('YE', closed='right').prod()
    msg = 'using SeriesGroupBy.prod'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = test_series.groupby(lambda x: x.year).agg(np.prod)
    expected.index = result.index
    tm.assert_series_equal(result, expected)