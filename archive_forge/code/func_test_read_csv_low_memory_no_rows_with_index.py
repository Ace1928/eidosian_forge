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
def test_read_csv_low_memory_no_rows_with_index(all_parsers):
    parser = all_parsers
    if not parser.low_memory:
        pytest.skip('This is a low-memory specific test')
    data = 'A,B,C\n1,1,1,2\n2,2,3,4\n3,3,4,5\n'
    if parser.engine == 'pyarrow':
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
        return
    result = parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
    expected = DataFrame(columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)