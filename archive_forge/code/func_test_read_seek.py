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
def test_read_seek(all_parsers):
    parser = all_parsers
    prefix = '### DATA\n'
    content = 'nkey,value\ntables,rectangular\n'
    with tm.ensure_clean() as path:
        Path(path).write_text(prefix + content, encoding='utf-8')
        with open(path, encoding='utf-8') as file:
            file.readline()
            actual = parser.read_csv(file)
        expected = parser.read_csv(StringIO(content))
    tm.assert_frame_equal(actual, expected)