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
@pytest.mark.parametrize('dtype,expected', [(True, Series(['2000-01-01'], dtype='datetime64[ns]')), (False, Series([946684800000]))])
def test_series_with_dtype_datetime(self, dtype, expected):
    s = Series(['2000-01-01'], dtype='datetime64[ns]')
    data = StringIO(s.to_json())
    result = read_json(data, typ='series', dtype=dtype)
    tm.assert_series_equal(result, expected)