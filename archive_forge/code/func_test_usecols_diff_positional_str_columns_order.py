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
@pytest.mark.parametrize('usecols', [['B', 'D'], ['D', 'B']])
def test_usecols_diff_positional_str_columns_order(self, read_ext, usecols, df_ref):
    expected = df_ref[['B', 'D']]
    expected.index = range(len(expected))
    result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1', usecols=usecols)
    tm.assert_frame_equal(result, expected)