from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_columns_casting():
    df = DataFrame({'a': [1.0, 2.0], 'b': [3, np.nan]})
    expected = df.copy()
    result = df.where(pd.notnull(df), None)
    tm.assert_frame_equal(expected, result)