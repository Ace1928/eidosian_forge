from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_empty_df_and_empty_cond_having_non_bool_dtypes(self):
    df = DataFrame(columns=['a'])
    cond = df
    assert (cond.dtypes == object).all()
    result = df.where(cond)
    tm.assert_frame_equal(result, df)