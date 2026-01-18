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
def test_reader_special_dtypes(self, request, engine, read_ext):
    xfail_datetimes_with_pyxlsb(engine, request)
    unit = get_exp_unit(read_ext, engine)
    expected = DataFrame.from_dict({'IntCol': [1, 2, -3, 4, 0], 'FloatCol': [1.25, 2.25, 1.83, 1.92, 5e-10], 'BoolCol': [True, False, True, True, False], 'StrCol': [1, 2, 3, 4, 5], 'Str2Col': ['a', 3, 'c', 'd', 'e'], 'DateCol': Index([datetime(2013, 10, 30), datetime(2013, 10, 31), datetime(1905, 1, 1), datetime(2013, 12, 14), datetime(2015, 3, 14)], dtype=f'M8[{unit}]')})
    basename = 'test_types'
    actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1')
    tm.assert_frame_equal(actual, expected)
    float_expected = expected.copy()
    float_expected.loc[float_expected.index[1], 'Str2Col'] = 3.0
    actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1')
    tm.assert_frame_equal(actual, float_expected)
    for icol, name in enumerate(expected.columns):
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1', index_col=icol)
        exp = expected.set_index(name)
        tm.assert_frame_equal(actual, exp)
    expected['StrCol'] = expected['StrCol'].apply(str)
    actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1', converters={'StrCol': str})
    tm.assert_frame_equal(actual, expected)