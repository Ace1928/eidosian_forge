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
def test_to_excel_timedelta(self, path):
    df = DataFrame(np.random.default_rng(2).integers(-10, 10, size=(20, 1)), columns=['A'], dtype=np.int64)
    expected = df.copy()
    df['new'] = df['A'].apply(lambda x: timedelta(seconds=x))
    expected['new'] = expected['A'].apply(lambda x: timedelta(seconds=x).total_seconds() / 86400)
    df.to_excel(path, sheet_name='test1')
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
    tm.assert_frame_equal(expected, recons)