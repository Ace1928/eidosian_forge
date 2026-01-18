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
def test_read_excel_multiindex_header_only(self, read_ext):
    mi_file = 'testmultiindex' + read_ext
    result = pd.read_excel(mi_file, sheet_name='index_col_none', header=[0, 1])
    exp_columns = MultiIndex.from_product([('A', 'B'), ('key', 'val')])
    expected = DataFrame([[1, 2, 3, 4]] * 2, columns=exp_columns)
    tm.assert_frame_equal(result, expected)