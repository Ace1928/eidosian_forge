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
def test_truncate_with_different_dtypes2(self):
    df = DataFrame({'text': ['some words'] + [None] * 9}, dtype=object)
    with option_context('display.max_rows', 8, 'display.max_columns', 3):
        result = str(df)
        assert 'None' in result
        assert 'NaN' not in result