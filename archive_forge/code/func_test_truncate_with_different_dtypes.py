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
@pytest.mark.parametrize('dtype', ['object', 'datetime64[us]'])
def test_truncate_with_different_dtypes(self, dtype):
    ser = Series([datetime(2012, 1, 1)] * 10 + [datetime(1012, 1, 2)] + [datetime(2012, 1, 3)] * 10, dtype=dtype)
    with option_context('display.max_rows', 8):
        result = str(ser)
    assert dtype in result