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
@pytest.mark.parametrize('header,expected', [(None, DataFrame([np.nan] * 4)), (0, DataFrame({'Unnamed: 0': [np.nan] * 3}))])
def test_read_one_empty_col_no_header(self, ext, header, expected):
    filename = 'no_header'
    df = DataFrame([['', 1, 100], ['', 2, 200], ['', 3, 300], ['', 4, 400]])
    with tm.ensure_clean(ext) as path:
        df.to_excel(path, sheet_name=filename, index=False, header=False)
        result = pd.read_excel(path, sheet_name=filename, usecols=[0], header=header)
    tm.assert_frame_equal(result, expected)