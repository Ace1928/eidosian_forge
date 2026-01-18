import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_float32_nanrep(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 4)).astype(np.float32))
    df[1] = np.nan
    with tm.ensure_clean('__tmp_to_csv_float32_nanrep__.csv') as path:
        df.to_csv(path, na_rep=999)
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            assert lines[1].split(',')[2] == '999'