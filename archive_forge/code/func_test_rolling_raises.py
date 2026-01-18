from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='axis not at all supported')
def test_rolling_raises():
    df = pd.DataFrame({'a': np.random.randn(25).cumsum(), 'b': np.random.randint(100, size=(25,))})
    ddf = dd.from_pandas(df, 3)
    pytest.raises(ValueError, lambda: ddf.rolling(1.5))
    pytest.raises(ValueError, lambda: ddf.rolling(-1))
    pytest.raises(ValueError, lambda: ddf.rolling(3, min_periods=1.2))
    pytest.raises(ValueError, lambda: ddf.rolling(3, min_periods=-2))
    axis_deprecated = pytest.warns(FutureWarning, match="'axis' keyword is deprecated")
    with axis_deprecated:
        pytest.raises(ValueError, lambda: ddf.rolling(3, axis=10))
    with axis_deprecated:
        pytest.raises(ValueError, lambda: ddf.rolling(3, axis='coulombs'))
    pytest.raises(NotImplementedError, lambda: ddf.rolling(100).mean().compute())