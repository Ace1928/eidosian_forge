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
def test_resample_nunique_preserves_column_level_names(unit):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=5, freq='D')).abs()
    df.index = df.index.as_unit(unit)
    df.columns = pd.MultiIndex.from_arrays([df.columns.tolist()] * 2, names=['lev0', 'lev1'])
    result = df.resample('1h').nunique()
    tm.assert_index_equal(df.columns, result.columns)