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
@pytest.mark.filterwarnings('ignore:Could not infer format:UserWarning')
@pytest.mark.parametrize('aware_val', [dtstr, Timestamp(dtstr)], ids=lambda x: type(x).__name__)
@pytest.mark.parametrize('naive_val', [dtstr[:-6], ts.tz_localize(None), ts.date(), ts.asm8, ts.value, float(ts.value)], ids=lambda x: type(x).__name__)
@pytest.mark.parametrize('naive_first', [True, False])
def test_to_datetime_mixed_awareness_mixed_types(aware_val, naive_val, naive_first):
    vals = [aware_val, naive_val, '']
    vec = vals
    if naive_first:
        vec = [naive_val, aware_val, '']
    both_strs = isinstance(aware_val, str) and isinstance(naive_val, str)
    has_numeric = isinstance(naive_val, (int, float))
    depr_msg = 'In a future version of pandas, parsing datetimes with mixed time zones'
    first_non_null = next((x for x in vec if x != ''))
    if not isinstance(first_non_null, str):
        msg = 'Cannot mix tz-aware with tz-naive values'
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = 'Tz-aware datetime.datetime cannot be converted to datetime64'
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
        else:
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
        to_datetime(vec, utc=True)
    elif has_numeric and vec.index(aware_val) < vec.index(naive_val):
        msg = "time data .* doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    elif both_strs and vec.index(aware_val) < vec.index(naive_val):
        msg = 'time data \\"2020-01-01 00:00\\" doesn\'t match format'
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    elif both_strs and vec.index(naive_val) < vec.index(aware_val):
        msg = 'unconverted data remains when parsing with format'
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    else:
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            to_datetime(vec)
        to_datetime(vec, utc=True)
    if both_strs:
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            to_datetime(vec, format='mixed')
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            msg = 'DatetimeIndex has mixed timezones'
            with pytest.raises(TypeError, match=msg):
                DatetimeIndex(vec)
    else:
        msg = 'Cannot mix tz-aware with tz-naive values'
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = 'Tz-aware datetime.datetime cannot be converted to datetime64'
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format='mixed')
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)
        else:
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format='mixed')
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)