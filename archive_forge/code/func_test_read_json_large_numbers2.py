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
def test_read_json_large_numbers2(self):
    json = '{"articleId": "1404366058080022500245"}'
    json = StringIO(json)
    result = read_json(json, typ='series')
    expected = Series(1.404366e+21, index=['articleId'])
    tm.assert_series_equal(result, expected)
    json = '{"0": {"articleId": "1404366058080022500245"}}'
    json = StringIO(json)
    result = read_json(json)
    expected = DataFrame(1.404366e+21, index=['articleId'], columns=[0])
    tm.assert_frame_equal(result, expected)