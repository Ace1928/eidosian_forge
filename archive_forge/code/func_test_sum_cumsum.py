import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('method', ['sum', 'cumsum'])
def test_sum_cumsum(self, df, method):
    expected_columns_numeric = Index(['int', 'float', 'category_int'])
    expected_columns = Index(['int', 'float', 'string', 'category_int', 'timedelta'])
    if method == 'cumsum':
        expected_columns = Index(['int', 'float', 'category_int', 'timedelta'])
    self._check(df, method, expected_columns, expected_columns_numeric)