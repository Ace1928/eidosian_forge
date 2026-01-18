import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('arr, expected_type, dtype', [(np.array([0, 1], dtype=np.int64), np.ndarray, 'int64'), (np.array(['a', 'b']), np.ndarray, 'object'), (pd.Categorical(['a', 'b']), pd.Categorical, 'category'), (pd.DatetimeIndex(['2017', '2018'], tz='US/Central'), DatetimeArray, 'datetime64[ns, US/Central]'), (pd.PeriodIndex([2018, 2019], freq='Y'), PeriodArray, pd.core.dtypes.dtypes.PeriodDtype('Y-DEC')), (pd.IntervalIndex.from_breaks([0, 1, 2]), IntervalArray, 'interval'), (pd.DatetimeIndex(['2017', '2018']), DatetimeArray, 'datetime64[ns]'), (pd.TimedeltaIndex([10 ** 10]), TimedeltaArray, 'm8[ns]')])
def test_values_consistent(arr, expected_type, dtype, using_infer_string):
    if using_infer_string and dtype == 'object':
        expected_type = ArrowStringArrayNumpySemantics
    l_values = Series(arr)._values
    r_values = pd.Index(arr)._values
    assert type(l_values) is expected_type
    assert type(l_values) is type(r_values)
    tm.assert_equal(l_values, r_values)