from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('data, expected', [(['3.50'], '0    3.50\ndtype: object'), ([1.2, '1.00'], '0     1.2\n1    1.00\ndtype: object'), ([np.nan], '0   NaN\ndtype: float64'), ([None], '0    None\ndtype: object'), (['3.50', np.nan], '0    3.50\n1     NaN\ndtype: object'), ([3.5, np.nan], '0    3.5\n1    NaN\ndtype: float64'), ([3.5, np.nan, '3.50'], '0     3.5\n1     NaN\n2    3.50\ndtype: object'), ([3.5, None, '3.50'], '0     3.5\n1    None\n2    3.50\ndtype: object')])
def test_repr_str_float_truncation(self, data, expected, using_infer_string):
    series = Series(data, dtype=object if '3.50' in data else None)
    result = repr(series)
    assert result == expected