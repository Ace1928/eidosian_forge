from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_compiled_regex_unicode(any_string_dtype):
    ser = Series([b'abcd,\xc3\xa0'.decode('utf-8')], dtype=any_string_dtype)
    expected = Series([b'abcd, \xc3\xa0'.decode('utf-8')], dtype=any_string_dtype)
    pat = re.compile('(?<=\\w),(?=\\w)', flags=re.UNICODE)
    with tm.maybe_produces_warning(PerformanceWarning, using_pyarrow(any_string_dtype)):
        result = ser.str.replace(pat, ', ', regex=True)
    tm.assert_series_equal(result, expected)