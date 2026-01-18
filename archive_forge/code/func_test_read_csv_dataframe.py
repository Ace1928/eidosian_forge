from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
def test_read_csv_dataframe(all_parsers, csv1):
    parser = all_parsers
    result = parser.read_csv(csv1, index_col=0, parse_dates=True)
    if parser.engine == 'pyarrow':
        result.index = result.index.as_unit('ns')
    expected = DataFrame([[0.980269, 3.685731, -0.364216805298, -1.159738], [1.047916, -0.041232, -0.16181208307, 0.212549], [0.498581, 0.731168, -0.537677223318, 1.34627], [1.120202, 1.567621, 0.00364077397681, 0.675253], [-0.487094, 0.571455, -1.6116394093, 0.103469], [0.836649, 0.246462, 0.588542635376, 1.062782], [-0.157161, 1.340307, 1.1957779562, -1.097007]], columns=['A', 'B', 'C', 'D'], index=Index([datetime(2000, 1, 3), datetime(2000, 1, 4), datetime(2000, 1, 5), datetime(2000, 1, 6), datetime(2000, 1, 7), datetime(2000, 1, 10), datetime(2000, 1, 11)], name='index'))
    tm.assert_frame_equal(result, expected)