from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_reader_seconds(self, request, engine, read_ext):
    xfail_datetimes_with_pyxlsb(engine, request)
    if engine == 'calamine' and read_ext == '.ods':
        request.applymarker(pytest.mark.xfail(reason='ODS file contains bad datetime (seconds as text)'))
    expected = DataFrame.from_dict({'Time': [time(1, 2, 3), time(2, 45, 56, 100000), time(4, 29, 49, 200000), time(6, 13, 42, 300000), time(7, 57, 35, 400000), time(9, 41, 28, 500000), time(11, 25, 21, 600000), time(13, 9, 14, 700000), time(14, 53, 7, 800000), time(16, 37, 0, 900000), time(18, 20, 54)]})
    actual = pd.read_excel('times_1900' + read_ext, sheet_name='Sheet1')
    tm.assert_frame_equal(actual, expected)
    actual = pd.read_excel('times_1904' + read_ext, sheet_name='Sheet1')
    tm.assert_frame_equal(actual, expected)