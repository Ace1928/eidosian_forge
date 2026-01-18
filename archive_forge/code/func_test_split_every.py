from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
@pytest.mark.parametrize('split_every', [None, 2, 10])
@pytest.mark.parametrize('npartitions', [2, 20])
def test_split_every(split_every, npartitions):
    df = pd.Series([1, 2, 3] * 1000)
    ddf = dd.from_pandas(df, npartitions=npartitions)
    approx = ddf.nunique_approx(split_every=split_every).compute(scheduler='sync')
    exact = len(df.drop_duplicates())
    assert abs(approx - exact) <= 2 or abs(approx - exact) / exact < 0.05