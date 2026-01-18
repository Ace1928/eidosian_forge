from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_mixed_object():
    ser = Series(['aBAD', np.nan, 'bBAD', True, datetime.today(), 'fooBAD', None, 1, 2.0])
    result = Series(ser).str.replace('BAD[_]*', '', regex=True)
    expected = Series(['a', np.nan, 'b', np.nan, np.nan, 'foo', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)