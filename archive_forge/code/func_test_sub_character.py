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
def test_sub_character(all_parsers, csv_dir_path):
    filename = os.path.join(csv_dir_path, 'sub_char.csv')
    expected = DataFrame([[1, 2, 3]], columns=['a', '\x1ab', 'c'])
    parser = all_parsers
    result = parser.read_csv(filename)
    tm.assert_frame_equal(result, expected)