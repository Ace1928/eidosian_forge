import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_coercion():
    df = DataFrame({'A': ['a', 'a', 'b', 'b'], 'B': [0, 1, 3, 4]})
    g = df.groupby('A')
    msg = 'using DataFrameGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = g.transform(np.mean)
    result = g.transform(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)