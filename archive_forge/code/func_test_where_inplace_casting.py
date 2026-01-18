from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@given(data=OPTIONAL_ONE_OF_ALL)
def test_where_inplace_casting(data):
    df = DataFrame({'a': data})
    df_copy = df.where(pd.notnull(df), None).copy()
    df.where(pd.notnull(df), None, inplace=True)
    tm.assert_equal(df, df_copy)