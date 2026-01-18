import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
@td.skip_if_windows
@pytest.mark.parametrize('arg_class', [Series, Index])
@pytest.mark.parametrize('utc', [True, False])
@pytest.mark.parametrize('tz', [None, 'US/Central'])
def test_to_datetime_arrow(self, tz, utc, arg_class):
    pa = pytest.importorskip('pyarrow')
    dti = date_range('1965-04-03', periods=19, freq='2W', tz=tz)
    dti = arg_class(dti)
    dti_arrow = dti.astype(pd.ArrowDtype(pa.timestamp(unit='ns', tz=tz)))
    result = to_datetime(dti_arrow, utc=utc)
    expected = to_datetime(dti, utc=utc).astype(pd.ArrowDtype(pa.timestamp(unit='ns', tz=tz if not utc else 'UTC')))
    if not utc and arg_class is not Series:
        assert result is dti_arrow
    if arg_class is Series:
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)