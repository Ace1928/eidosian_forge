from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_skip_rows_skip_all(all_parsers):
    parser = all_parsers
    data = 'a\n1\n2\n3\n4\n5'
    msg = 'No columns to parse from file'
    with pytest.raises(EmptyDataError, match=msg):
        parser.read_csv(StringIO(data), skiprows=lambda x: True)