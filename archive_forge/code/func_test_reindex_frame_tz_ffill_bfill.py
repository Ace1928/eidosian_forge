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
@pytest.mark.parametrize('method, exp_values', [('ffill', [0, 1, 2, 3]), ('bfill', [1.0, 2.0, 3.0, np.nan])])
def test_reindex_frame_tz_ffill_bfill(self, frame_or_series, method, exp_values):
    obj = frame_or_series([0, 1, 2, 3], index=date_range('2020-01-01 00:00:00', periods=4, freq='h', tz='UTC'))
    new_index = date_range('2020-01-01 00:01:00', periods=4, freq='h', tz='UTC')
    result = obj.reindex(new_index, method=method, tolerance=pd.Timedelta('1 hour'))
    expected = frame_or_series(exp_values, index=new_index)
    tm.assert_equal(result, expected)