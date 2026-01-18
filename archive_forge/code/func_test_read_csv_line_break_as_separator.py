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
@pytest.mark.parametrize('kwargs', [{'delimiter': '\n'}, {'sep': '\n'}])
def test_read_csv_line_break_as_separator(kwargs, all_parsers):
    parser = all_parsers
    data = 'a,b,c\n1,2,3\n    '
    msg = 'Specified \\\\n as separator or delimiter. This forces the python engine which does not accept a line terminator. Hence it is not allowed to use the line terminator as separator.'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)