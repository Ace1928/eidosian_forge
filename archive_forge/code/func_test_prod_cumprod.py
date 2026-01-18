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
@pytest.mark.parametrize('method', ['prod', 'cumprod'])
def test_prod_cumprod(self, df, method):
    expected_columns = Index(['int', 'float', 'category_int'])
    expected_columns_numeric = expected_columns
    self._check(df, method, expected_columns, expected_columns_numeric)