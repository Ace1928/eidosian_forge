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
@skip_pyarrow
def test_whitespace_lines(all_parsers):
    parser = all_parsers
    data = '\n\n\t  \t\t\n\t\nA,B,C\n\t    1,2.,4.\n5.,NaN,10.0\n'
    expected = DataFrame([[1, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)