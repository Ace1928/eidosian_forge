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
@pytest.mark.parametrize('usecols', [None, [0, 1], ['a', 'b']])
def test_uneven_lines_with_usecols(all_parsers, usecols):
    parser = all_parsers
    data = 'a,b,c\n0,1,2\n3,4,5,6,7\n8,9,10'
    if usecols is None:
        msg = 'Expected \\d+ fields in line \\d+, saw \\d+'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
    else:
        expected = DataFrame({'a': [0, 3, 8], 'b': [1, 4, 9]})
        result = parser.read_csv(StringIO(data), usecols=usecols)
        tm.assert_frame_equal(result, expected)