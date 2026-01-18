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
def test_read_filepath_or_buffer(all_parsers):
    parser = all_parsers
    with pytest.raises(TypeError, match='Expected file path name or file-like'):
        parser.read_csv(filepath_or_buffer=b'input')