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
def test_series_describe_multikey():
    ts = tm.makeTimeSeries()
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.describe()
    tm.assert_series_equal(result['mean'], grouped.mean(), check_names=False)
    tm.assert_series_equal(result['std'], grouped.std(), check_names=False)
    tm.assert_series_equal(result['min'], grouped.min(), check_names=False)