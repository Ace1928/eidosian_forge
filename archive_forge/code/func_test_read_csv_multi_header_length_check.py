from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_read_csv_multi_header_length_check(all_parsers):
    parser = all_parsers
    case = 'row11,row12,row13\nrow21,row22, row23\nrow31,row32\n'
    with pytest.raises(ParserError, match='Header rows must have an equal number of columns.'):
        parser.read_csv(StringIO(case), header=[0, 2])