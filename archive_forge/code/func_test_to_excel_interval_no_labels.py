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
def test_to_excel_interval_no_labels(self, path, using_infer_string):
    df = DataFrame(np.random.default_rng(2).integers(-10, 10, size=(20, 1)), dtype=np.int64)
    expected = df.copy()
    df['new'] = pd.cut(df[0], 10)
    expected['new'] = pd.cut(expected[0], 10).astype(str if not using_infer_string else 'string[pyarrow_numpy]')
    df.to_excel(path, sheet_name='test1')
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=0)
    tm.assert_frame_equal(expected, recons)