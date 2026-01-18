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
def test_read_excel_multiindex_empty_level(self, ext):
    with tm.ensure_clean(ext) as path:
        df = DataFrame({('One', 'x'): {0: 1}, ('Two', 'X'): {0: 3}, ('Two', 'Y'): {0: 7}, ('Zero', ''): {0: 0}})
        expected = DataFrame({('One', 'x'): {0: 1}, ('Two', 'X'): {0: 3}, ('Two', 'Y'): {0: 7}, ('Zero', 'Unnamed: 4_level_1'): {0: 0}})
        df.to_excel(path)
        actual = pd.read_excel(path, header=[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)
        df = DataFrame({('Beg', ''): {0: 0}, ('Middle', 'x'): {0: 1}, ('Tail', 'X'): {0: 3}, ('Tail', 'Y'): {0: 7}})
        expected = DataFrame({('Beg', 'Unnamed: 1_level_1'): {0: 0}, ('Middle', 'x'): {0: 1}, ('Tail', 'X'): {0: 3}, ('Tail', 'Y'): {0: 7}})
        df.to_excel(path)
        actual = pd.read_excel(path, header=[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)