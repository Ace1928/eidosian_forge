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
def test_read_timezone_information(self):
    result = read_json(StringIO('{"2019-01-01T11:00:00.000Z":88}'), typ='series', orient='index')
    exp_dti = DatetimeIndex(['2019-01-01 11:00:00'], dtype='M8[ns, UTC]')
    expected = Series([88], index=exp_dti)
    tm.assert_series_equal(result, expected)