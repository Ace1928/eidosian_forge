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
@pytest.mark.parametrize('listlike,do_caching', [([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], False), ([1, 1, 1, 1, 4, 5, 6, 7, 8, 9], True)])
def test_should_cache(self, listlike, do_caching):
    assert tools.should_cache(listlike, check_count=len(listlike), unique_share=0.7) == do_caching