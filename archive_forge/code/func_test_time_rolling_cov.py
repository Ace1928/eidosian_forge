from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('window', ['1s', '2s', '3s', pd.offsets.Second(5)])
def test_time_rolling_cov(window):
    prolling = ts.drop('a', axis=1).rolling(window)
    drolling = dts.drop('a', axis=1).rolling(window)
    assert_eq(prolling.cov(), drolling.cov())
    prolling = ts.b.rolling(window)
    drolling = dts.b.rolling(window)
    assert_eq(prolling.cov(), drolling.cov())