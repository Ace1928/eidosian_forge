from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def test_time_rolling_constructor():
    result = dts.rolling('4s')
    assert result.window == '4s'
    assert result.min_periods is None
    assert result.win_type is None
    if not DASK_EXPR_ENABLED:
        assert result._win_type == 'freq'