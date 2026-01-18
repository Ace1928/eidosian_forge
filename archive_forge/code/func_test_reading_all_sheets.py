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
def test_reading_all_sheets(self, read_ext):
    basename = 'test_multisheet'
    dfs = pd.read_excel(basename + read_ext, sheet_name=None)
    expected_keys = ['Charlie', 'Alpha', 'Beta']
    tm.assert_contains_all(expected_keys, dfs.keys())
    assert expected_keys == list(dfs.keys())