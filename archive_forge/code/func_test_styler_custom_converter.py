import contextlib
import time
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.excel import ExcelWriter
from pandas.io.formats.excel import ExcelFormatter
def test_styler_custom_converter():
    openpyxl = pytest.importorskip('openpyxl')

    def custom_converter(css):
        return {'font': {'color': {'rgb': '111222'}}}
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 1)))
    styler = df.style.map(lambda x: 'color: #888999')
    with tm.ensure_clean('.xlsx') as path:
        with ExcelWriter(path, engine='openpyxl') as writer:
            ExcelFormatter(styler, style_converter=custom_converter).write(writer, sheet_name='custom')
        with contextlib.closing(openpyxl.load_workbook(path)) as wb:
            assert wb['custom'].cell(2, 2).font.color.value == '00111222'