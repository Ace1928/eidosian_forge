import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_bug():
    df = DataFrame({'A': Timestamp('20130101'), 'B': np.arange(5)})
    result = df.groupby('A')['B'].transform(lambda x: x.rank(ascending=False))
    expected = Series(np.arange(5, 0, step=-1), name='B', dtype='float64')
    tm.assert_series_equal(result, expected)