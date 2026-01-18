from __future__ import annotations
import string
from typing import cast
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base
def test_searchsorted_with_na_raises(data_for_sorting, as_series):
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])
    arr[-1] = pd.NA
    if as_series:
        arr = pd.Series(arr)
    msg = 'searchsorted requires array to be sorted, which is impossible with NAs present.'
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)