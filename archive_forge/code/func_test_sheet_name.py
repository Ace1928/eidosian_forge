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
def test_sheet_name(self, request, engine, read_ext, df_ref):
    xfail_datetimes_with_pyxlsb(engine, request)
    expected = df_ref
    adjust_expected(expected, read_ext, engine)
    filename = 'test1'
    sheet_name = 'Sheet1'
    with pd.ExcelFile(filename + read_ext) as excel:
        df1_parse = excel.parse(sheet_name=sheet_name, index_col=0)
    with pd.ExcelFile(filename + read_ext) as excel:
        df2_parse = excel.parse(index_col=0, sheet_name=sheet_name)
    tm.assert_frame_equal(df1_parse, expected)
    tm.assert_frame_equal(df2_parse, expected)