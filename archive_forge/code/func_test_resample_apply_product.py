from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
@pytest.mark.parametrize('duplicates', [True, False])
def test_resample_apply_product(duplicates, unit):
    index = date_range(start='2012-01-31', freq='ME', periods=12).as_unit(unit)
    ts = Series(range(12), index=index)
    df = DataFrame({'A': ts, 'B': ts + 2})
    if duplicates:
        df.columns = ['A', 'A']
    msg = 'using DatetimeIndexResampler.prod'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.resample('QE').apply(np.prod)
    expected = DataFrame(np.array([[0, 24], [60, 210], [336, 720], [990, 1716]], dtype=np.int64), index=DatetimeIndex(['2012-03-31', '2012-06-30', '2012-09-30', '2012-12-31'], freq='QE-DEC').as_unit(unit), columns=df.columns)
    tm.assert_frame_equal(result, expected)