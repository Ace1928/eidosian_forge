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
def test_to_csv_unicode_index_col(self):
    buf = StringIO('')
    df = DataFrame([['א', 'd2', 'd3', 'd4'], ['a1', 'a2', 'a3', 'a4']], columns=['א', 'ב', 'ג', 'ד'], index=['א', 'ב'])
    df.to_csv(buf, encoding='UTF-8')
    buf.seek(0)
    df2 = read_csv(buf, index_col=0, encoding='UTF-8')
    tm.assert_frame_equal(df, df2)