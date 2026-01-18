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
def test_to_excel_multiindex_dates(self, merge_cells, path):
    unit = get_exp_unit(path)
    tsframe = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD')), index=date_range('2000-01-01', periods=5, freq='B'))
    tsframe.index = MultiIndex.from_arrays([tsframe.index.as_unit(unit), np.arange(len(tsframe.index), dtype=np.int64)], names=['time', 'foo'])
    tsframe.to_excel(path, sheet_name='test1', merge_cells=merge_cells)
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=[0, 1])
    tm.assert_frame_equal(tsframe, recons)
    assert recons.index.names == ('time', 'foo')