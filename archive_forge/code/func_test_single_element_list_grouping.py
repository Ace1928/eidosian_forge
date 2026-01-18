from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
@pytest.mark.parametrize('selection', [None, 'a', ['a']])
def test_single_element_list_grouping(selection):
    df = DataFrame({'a': [1, 2], 'b': [np.nan, 5], 'c': [np.nan, 2]}, index=['x', 'y'])
    grouped = df.groupby(['a']) if selection is None else df.groupby(['a'])[selection]
    result = [key for key, _ in grouped]
    expected = [(1,), (2,)]
    assert result == expected