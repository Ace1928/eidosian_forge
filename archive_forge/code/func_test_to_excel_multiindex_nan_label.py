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
def test_to_excel_multiindex_nan_label(self, merge_cells, path):
    df = DataFrame({'A': [None, 2, 3], 'B': [10, 20, 30], 'C': np.random.default_rng(2).random(3)})
    df = df.set_index(['A', 'B'])
    df.to_excel(path, merge_cells=merge_cells)
    df1 = pd.read_excel(path, index_col=[0, 1])
    tm.assert_frame_equal(df, df1)