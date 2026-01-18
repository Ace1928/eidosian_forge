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
def test_set_column_names_in_parameter(self, ext):
    refdf = DataFrame([[1, 'foo'], [2, 'bar'], [3, 'baz']], columns=['a', 'b'])
    with tm.ensure_clean(ext) as pth:
        with ExcelWriter(pth) as writer:
            refdf.to_excel(writer, sheet_name='Data_no_head', header=False, index=False)
            refdf.to_excel(writer, sheet_name='Data_with_head', index=False)
        refdf.columns = ['A', 'B']
        with ExcelFile(pth) as reader:
            xlsdf_no_head = pd.read_excel(reader, sheet_name='Data_no_head', header=None, names=['A', 'B'])
            xlsdf_with_head = pd.read_excel(reader, sheet_name='Data_with_head', index_col=None, names=['A', 'B'])
        tm.assert_frame_equal(xlsdf_no_head, refdf)
        tm.assert_frame_equal(xlsdf_with_head, refdf)