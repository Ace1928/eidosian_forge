from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ix_data, exp_data', [([(pd.NaT, 1), (pd.NaT, 2)], {'a': [pd.NaT, pd.NaT], 'b': [1, 2], 'x': [11, 12]}), ([(pd.NaT, 1), (Timestamp('2020-01-01'), 2)], {'a': [pd.NaT, Timestamp('2020-01-01')], 'b': [1, 2], 'x': [11, 12]}), ([(pd.NaT, 1), (pd.Timedelta(123, 'd'), 2)], {'a': [pd.NaT, pd.Timedelta(123, 'd')], 'b': [1, 2], 'x': [11, 12]})])
def test_reset_index_nat_multiindex(self, ix_data, exp_data):
    ix = MultiIndex.from_tuples(ix_data, names=['a', 'b'])
    result = DataFrame({'x': [11, 12]}, index=ix)
    result = result.reset_index()
    expected = DataFrame(exp_data)
    tm.assert_frame_equal(result, expected)