from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_parse_ragged_csv(c_parser_only):
    parser = c_parser_only
    data = '1,2,3\n1,2,3,4\n1,2,3,4,5\n1,2\n1,2,3,4'
    nice_data = '1,2,3,,\n1,2,3,4,\n1,2,3,4,5\n1,2,,,\n1,2,3,4,'
    result = parser.read_csv(StringIO(data), header=None, names=['a', 'b', 'c', 'd', 'e'])
    expected = parser.read_csv(StringIO(nice_data), header=None, names=['a', 'b', 'c', 'd', 'e'])
    tm.assert_frame_equal(result, expected)
    data = '1,2\n3,4,5'
    result = parser.read_csv(StringIO(data), header=None, names=range(50))
    expected = parser.read_csv(StringIO(data), header=None, names=range(3)).reindex(columns=range(50))
    tm.assert_frame_equal(result, expected)