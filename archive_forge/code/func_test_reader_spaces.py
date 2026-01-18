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
def test_reader_spaces(self, read_ext):
    basename = 'test_spaces'
    actual = pd.read_excel(basename + read_ext)
    expected = DataFrame({'testcol': ['this is great', '4    spaces', '1 trailing ', ' 1 leading', '2  spaces  multiple  times']})
    tm.assert_frame_equal(actual, expected)