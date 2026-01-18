from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('df', [pd.DataFrame({'a': [-1, 1]}), pd.DataFrame({'a': [False, True]}), pd.DataFrame({'a': pd.Series(pd.to_timedelta([-1, 1]))})])
def test_pos_numeric(self, df):
    tm.assert_frame_equal(+df, df)
    tm.assert_series_equal(+df['a'], df['a'])