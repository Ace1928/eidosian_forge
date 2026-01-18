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
def test_usecols_list(self, request, engine, read_ext, df_ref):
    xfail_datetimes_with_pyxlsb(engine, request)
    expected = df_ref[['B', 'C']]
    adjust_expected(expected, read_ext, engine)
    df1 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', index_col=0, usecols=[0, 2, 3])
    df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows=[1], index_col=0, usecols=[0, 2, 3])
    tm.assert_frame_equal(df1, expected)
    tm.assert_frame_equal(df2, expected)