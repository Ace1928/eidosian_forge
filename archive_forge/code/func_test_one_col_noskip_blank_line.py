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
def test_one_col_noskip_blank_line(self, read_ext):
    file_name = 'one_col_blank_line' + read_ext
    data = [0.5, np.nan, 1, 2]
    expected = DataFrame(data, columns=['numbers'])
    result = pd.read_excel(file_name)
    tm.assert_frame_equal(result, expected)