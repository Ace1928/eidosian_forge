from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('window', [1, 2, 4, 5])
@pytest.mark.parametrize('center', [True, False])
def test_rolling_cov(window, center):
    prolling = df.drop('a', axis=1).rolling(window, center=center)
    drolling = ddf.drop('a', axis=1).rolling(window, center=center)
    assert_eq(prolling.cov(), drolling.cov())
    prolling = df.b.rolling(window, center=center)
    drolling = ddf.b.rolling(window, center=center)
    assert_eq(prolling.cov(), drolling.cov())