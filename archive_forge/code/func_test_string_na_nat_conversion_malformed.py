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
def test_string_na_nat_conversion_malformed(self, cache):
    malformed = np.array(['1/100/2000', np.nan], dtype=object)
    msg = 'Unknown datetime string format'
    with pytest.raises(ValueError, match=msg):
        to_datetime(malformed, errors='raise', cache=cache)
    result = to_datetime(malformed, errors='ignore', cache=cache)
    expected = Index(malformed, dtype=object)
    tm.assert_index_equal(result, expected)
    with pytest.raises(ValueError, match=msg):
        to_datetime(malformed, errors='raise', cache=cache)