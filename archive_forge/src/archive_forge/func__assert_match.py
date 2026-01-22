import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs import NaT
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
import pandas as pd
def _assert_match(result_fill_value, expected_fill_value):
    res_type = type(result_fill_value)
    ex_type = type(expected_fill_value)
    if hasattr(result_fill_value, 'dtype'):
        assert result_fill_value.dtype.kind == expected_fill_value.dtype.kind
        assert result_fill_value.dtype.itemsize == expected_fill_value.dtype.itemsize
    else:
        assert res_type == ex_type or res_type.__name__ == ex_type.__name__
    match_value = result_fill_value == expected_fill_value
    if match_value is pd.NA:
        match_value = False
    match_missing = isna(result_fill_value) and isna(expected_fill_value)
    assert match_value or match_missing