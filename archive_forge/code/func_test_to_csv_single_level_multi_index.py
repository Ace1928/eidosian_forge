import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_single_level_multi_index(self):
    index = Index([(1,), (2,), (3,)])
    df = DataFrame([[1, 2, 3]], columns=index)
    df = df.reindex(columns=[(1,), (3,)])
    expected = ',1,3\n0,1,3\n'
    result = df.to_csv(lineterminator='\n')
    tm.assert_almost_equal(result, expected)