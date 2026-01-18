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
def test_to_excel_multiindex_cols(self, merge_cells, frame, path):
    arrays = np.arange(len(frame.index) * 2, dtype=np.int64).reshape(2, -1)
    new_index = MultiIndex.from_arrays(arrays, names=['first', 'second'])
    frame.index = new_index
    new_cols_index = MultiIndex.from_tuples([(40, 1), (40, 2), (50, 1), (50, 2)])
    frame.columns = new_cols_index
    header = [0, 1]
    if not merge_cells:
        header = 0
    frame.to_excel(path, sheet_name='test1', merge_cells=merge_cells)
    with ExcelFile(path) as reader:
        df = pd.read_excel(reader, sheet_name='test1', header=header, index_col=[0, 1])
    if not merge_cells:
        fm = frame.columns._format_multi(sparsify=False, include_names=False)
        frame.columns = ['.'.join(map(str, q)) for q in zip(*fm)]
    tm.assert_frame_equal(frame, df)