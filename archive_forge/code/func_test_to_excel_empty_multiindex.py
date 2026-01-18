from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
def test_to_excel_empty_multiindex(self, path):
    expected = DataFrame([], columns=[0, 1, 2])
    df = DataFrame([], index=MultiIndex.from_tuples([], names=[0, 1]), columns=[2])
    df.to_excel(path, sheet_name='test1')
    with ExcelFile(path) as reader:
        result = pd.read_excel(reader, sheet_name='test1')
    tm.assert_frame_equal(result, expected, check_index_type=False, check_dtype=False)