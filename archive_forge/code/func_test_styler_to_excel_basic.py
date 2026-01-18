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
@pytest.mark.parametrize('engine', ['xlsxwriter', 'openpyxl'])
@pytest.mark.parametrize('css, attrs, expected', shared_style_params)
def test_styler_to_excel_basic(engine, css, attrs, expected):
    pytest.importorskip(engine)
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 1)))
    styler = df.style.map(lambda x: css)
    with tm.ensure_clean('.xlsx') as path:
        with ExcelWriter(path, engine=engine) as writer:
            df.to_excel(writer, sheet_name='dataframe')
            styler.to_excel(writer, sheet_name='styled')
        openpyxl = pytest.importorskip('openpyxl')
        with contextlib.closing(openpyxl.load_workbook(path)) as wb:
            u_cell, s_cell = (wb['dataframe'].cell(2, 2), wb['styled'].cell(2, 2))
        for attr in attrs:
            u_cell, s_cell = (getattr(u_cell, attr, None), getattr(s_cell, attr))
        if isinstance(expected, dict):
            assert u_cell is None or u_cell != expected[engine]
            assert s_cell == expected[engine]
        else:
            assert u_cell is None or u_cell != expected
            assert s_cell == expected