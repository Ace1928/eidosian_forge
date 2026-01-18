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
def test_to_excel_pos_args_deprecation(self):
    df = DataFrame({'a': [1, 2, 3]})
    msg = "Starting with pandas version 3.0 all arguments of to_excel except for the argument 'excel_writer' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buf = BytesIO()
        writer = ExcelWriter(buf)
        df.to_excel(writer, 'Sheet_name_1')