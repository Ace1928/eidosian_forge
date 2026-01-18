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
def test_to_csv_float_format(self):
    df = DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
    with tm.ensure_clean() as filename:
        df.to_csv(filename, float_format='%.2f')
        rs = read_csv(filename, index_col=0)
        xp = DataFrame([[0.12, 0.23, 0.57], [12.32, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
        tm.assert_frame_equal(rs, xp)