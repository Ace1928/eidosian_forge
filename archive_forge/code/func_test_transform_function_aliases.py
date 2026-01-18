import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_function_aliases(df):
    result = df.groupby('A').transform('mean', numeric_only=True)
    msg = 'using DataFrameGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby('A')[['C', 'D']].transform(np.mean)
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A')['C'].transform('mean')
    msg = 'using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby('A')['C'].transform(np.mean)
    tm.assert_series_equal(result, expected)