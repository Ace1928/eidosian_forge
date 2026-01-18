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
def test_render_as_column_name(self, path):
    df = DataFrame({'render': [1, 2], 'data': [3, 4]})
    df.to_excel(path, sheet_name='Sheet1')
    read = pd.read_excel(path, 'Sheet1', index_col=0)
    expected = df
    tm.assert_frame_equal(read, expected)