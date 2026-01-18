from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_complex(self):
    expected = DataFrame([[1 + 1j, 2], [np.nan, 4 + 1j]], columns=['a', 'b'])
    df = DataFrame([[1 + 1j, 2], [5 + 1j, 4 + 1j]], columns=['a', 'b'])
    df[df.abs() >= 5] = np.nan
    tm.assert_frame_equal(df, expected)