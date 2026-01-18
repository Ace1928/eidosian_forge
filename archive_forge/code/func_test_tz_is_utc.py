import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('ts', [Timestamp('2013-01-10 05:00:00Z'), Timestamp('2013-01-10 00:00:00', tz='US/Eastern'), Timestamp('2013-01-10 00:00:00-0500')])
def test_tz_is_utc(self, ts):
    exp = '"2013-01-10T05:00:00.000Z"'
    assert ujson_dumps(ts, iso_dates=True) == exp
    dt = ts.to_pydatetime()
    assert ujson_dumps(dt, iso_dates=True) == exp