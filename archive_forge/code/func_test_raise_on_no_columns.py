import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
@skip_pyarrow
@pytest.mark.parametrize('nrows', [0, 1, 2, 3, 4, 5])
def test_raise_on_no_columns(all_parsers, nrows):
    parser = all_parsers
    data = '\n' * nrows
    msg = 'No columns to parse from file'
    with pytest.raises(EmptyDataError, match=msg):
        parser.read_csv(StringIO(data))