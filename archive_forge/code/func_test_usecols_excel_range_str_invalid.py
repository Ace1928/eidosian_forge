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
def test_usecols_excel_range_str_invalid(self, read_ext):
    msg = 'Invalid column name: E1'
    with pytest.raises(ValueError, match=msg):
        pd.read_excel('test1' + read_ext, sheet_name='Sheet1', usecols='D:E1')