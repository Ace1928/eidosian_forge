import contextlib
from pathlib import Path
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._openpyxl import OpenpyxlReader
def test_to_excel_with_openpyxl_engine(ext):
    with tm.ensure_clean(ext) as filename:
        df1 = DataFrame({'A': np.linspace(1, 10, 10)})
        df2 = DataFrame({'B': np.linspace(1, 20, 10)})
        df = pd.concat([df1, df2], axis=1)
        styled = df.style.map(lambda val: f'color: {('red' if val < 0 else 'black')}').highlight_max()
        styled.to_excel(filename, engine='openpyxl')