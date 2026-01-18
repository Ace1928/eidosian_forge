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
@xfail_pyarrow
def test_unnamed_columns(all_parsers):
    data = 'A,B,C,,\n1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15\n'
    parser = all_parsers
    expected = DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=np.int64, columns=['A', 'B', 'C', 'Unnamed: 3', 'Unnamed: 4'])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)