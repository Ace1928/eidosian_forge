from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('df', dfs)
def test_where_last_n(df):
    sol_rowindex = np.array([[[4, 3, 1, 0, -1, -1], [14, 13, 12, 11, 10, -1]], [[9, 8, 7, 6, 5, -1], [19, 18, 17, 16, 15, -1]]])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds.last_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.last('plusminus'))).data)
        agg = c.points(df, 'x', 'y', ds.where(ds.last_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.last('plusminus'), 'reverse')).data)