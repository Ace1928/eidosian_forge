from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def test_rolling_partition_size():
    df = pd.DataFrame(np.random.randn(50, 2))
    ddf = dd.from_pandas(df, npartitions=5)
    for obj, dobj in [(df, ddf), (df[0], ddf[0])]:
        assert_eq(obj.rolling(10).mean(), dobj.rolling(10).mean())
        assert_eq(obj.rolling(11).mean(), dobj.rolling(11).mean())
        with pytest.raises(NotImplementedError):
            dobj.rolling(12).mean().compute()