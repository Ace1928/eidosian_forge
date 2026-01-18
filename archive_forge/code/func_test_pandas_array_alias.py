from __future__ import annotations
import pytest
import pandas as pd
from pandas import api
import pandas._testing as tm
from pandas.api import (
def test_pandas_array_alias():
    msg = 'PandasArray has been renamed NumpyExtensionArray'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = pd.arrays.PandasArray
    assert res is pd.arrays.NumpyExtensionArray