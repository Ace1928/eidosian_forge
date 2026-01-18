from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('method,args,check_less_precise', rolling_method_args_check_less_precise)
@pytest.mark.parametrize('window', ['1s', '2s', '3s', pd.offsets.Second(5)])
def test_time_rolling_methods(method, args, window, check_less_precise):
    if check_less_precise:
        check_less_precise = {'atol': 0.001, 'rtol': 0.001}
    else:
        check_less_precise = {}
    if method == 'apply':
        kwargs = {'raw': False}
    else:
        kwargs = {}
    prolling = ts.rolling(window)
    drolling = dts.rolling(window)
    assert_eq(getattr(prolling, method)(*args, **kwargs), getattr(drolling, method)(*args, **kwargs), **check_less_precise)
    prolling = ts.a.rolling(window)
    drolling = dts.a.rolling(window)
    assert_eq(getattr(prolling, method)(*args, **kwargs), getattr(drolling, method)(*args, **kwargs), **check_less_precise)