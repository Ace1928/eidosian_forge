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
def test_nrows_skipfooter_errors(all_parsers):
    msg = "'skipfooter' not supported with 'nrows'"
    data = 'a\n1\n2\n3\n4\n5\n6'
    parser = all_parsers
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=1, nrows=5)