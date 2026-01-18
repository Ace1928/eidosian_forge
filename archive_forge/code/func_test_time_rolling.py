from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('before, after', [('6s', '6s'), ('2s', '2s'), ('6s', '2s')])
def test_time_rolling(before, after):
    window = before
    expected = dts.compute().rolling(window).count()
    result = dts.map_overlap(lambda x: x.rolling(window).count(), before, after)
    assert_eq(result, expected)
    before = pd.Timedelta(before)
    after = pd.Timedelta(after)
    result = dts.map_overlap(lambda x: x.rolling(window).count(), before, after)
    assert_eq(result, expected)