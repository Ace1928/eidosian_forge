from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_inplace_no_other():
    df = DataFrame({'a': [1.0, 2.0], 'b': ['x', 'y']})
    cond = DataFrame({'a': [True, False], 'b': [False, True]})
    df.where(cond, inplace=True)
    expected = DataFrame({'a': [1, np.nan], 'b': [np.nan, 'y']})
    tm.assert_frame_equal(df, expected)