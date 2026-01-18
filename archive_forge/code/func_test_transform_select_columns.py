import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_select_columns(df):
    f = lambda x: x.mean()
    result = df.groupby('A')[['C', 'D']].transform(f)
    selection = df[['C', 'D']]
    expected = selection.groupby(df['A']).transform(f)
    tm.assert_frame_equal(result, expected)