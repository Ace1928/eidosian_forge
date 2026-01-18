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
def test_roundtrip_indexlabels(self, merge_cells, frame, path):
    frame = frame.copy()
    frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
    frame.to_excel(path, sheet_name='test1')
    frame.to_excel(path, sheet_name='test1', columns=['A', 'B'])
    frame.to_excel(path, sheet_name='test1', header=False)
    frame.to_excel(path, sheet_name='test1', index=False)
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
    df.to_excel(path, sheet_name='test1', index_label=['test'], merge_cells=merge_cells)
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=0).astype(np.int64)
    df.index.names = ['test']
    assert df.index.names == recons.index.names
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
    df.to_excel(path, sheet_name='test1', index_label=['test', 'dummy', 'dummy2'], merge_cells=merge_cells)
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=0).astype(np.int64)
    df.index.names = ['test']
    assert df.index.names == recons.index.names
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2))) >= 0
    df.to_excel(path, sheet_name='test1', index_label='test', merge_cells=merge_cells)
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=0).astype(np.int64)
    df.index.names = ['test']
    tm.assert_frame_equal(df, recons.astype(bool))
    frame.to_excel(path, sheet_name='test1', columns=['A', 'B', 'C', 'D'], index=False, merge_cells=merge_cells)
    df = frame.copy()
    df = df.set_index(['A', 'B'])
    with ExcelFile(path) as reader:
        recons = pd.read_excel(reader, sheet_name='test1', index_col=[0, 1])
    tm.assert_frame_equal(df, recons)