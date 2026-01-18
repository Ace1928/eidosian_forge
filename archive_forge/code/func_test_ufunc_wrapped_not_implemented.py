from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
def test_ufunc_wrapped_not_implemented():
    s = pd.Series(np.random.randint(1, 100, size=20), index=list('abcdefghijklmnopqrst'))
    ds = dd.from_pandas(s, 3)
    with pytest.raises(NotImplementedError, match='`repeat` is not implemented'):
        np.repeat(ds, 10)
    df = pd.DataFrame({'A': np.random.randint(1, 100, size=20), 'B': np.random.randint(1, 100, size=20), 'C': np.abs(np.random.randn(20))}, index=list('abcdefghijklmnopqrst'))
    ddf = dd.from_pandas(df, 3)
    with pytest.raises(NotImplementedError, match='`repeat` is not implemented'):
        np.repeat(ddf, 10)