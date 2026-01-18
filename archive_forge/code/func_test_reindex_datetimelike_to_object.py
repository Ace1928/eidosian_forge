from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('dtype', ['m8[ns]', 'M8[ns]'])
def test_reindex_datetimelike_to_object(self, dtype):
    mi = MultiIndex.from_product([list('ABCDE'), range(2)])
    dti = date_range('2016-01-01', periods=10)
    fv = np.timedelta64('NaT', 'ns')
    if dtype == 'm8[ns]':
        dti = dti - dti[0]
        fv = np.datetime64('NaT', 'ns')
    ser = Series(dti, index=mi)
    ser[::3] = pd.NaT
    df = ser.unstack()
    index = df.index.append(Index([1]))
    columns = df.columns.append(Index(['foo']))
    res = df.reindex(index=index, columns=columns, fill_value=fv)
    expected = DataFrame({0: df[0].tolist() + [fv], 1: df[1].tolist() + [fv], 'foo': np.array(['NaT'] * 6, dtype=fv.dtype)}, index=index)
    assert (res.dtypes[[0, 1]] == object).all()
    assert res.iloc[0, 0] is pd.NaT
    assert res.iloc[-1, 0] is fv
    assert res.iloc[-1, 1] is fv
    tm.assert_frame_equal(res, expected)