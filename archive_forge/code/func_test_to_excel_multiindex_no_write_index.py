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
def test_to_excel_multiindex_no_write_index(self, path):
    frame1 = DataFrame({'a': [10, 20], 'b': [30, 40], 'c': [50, 60]})
    frame2 = frame1.copy()
    multi_index = MultiIndex.from_tuples([(70, 80), (90, 100)])
    frame2.index = multi_index
    frame2.to_excel(path, sheet_name='test1', index=False)
    with ExcelFile(path) as reader:
        frame3 = pd.read_excel(reader, sheet_name='test1')
    tm.assert_frame_equal(frame1, frame3)