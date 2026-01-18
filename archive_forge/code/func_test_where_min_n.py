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
def test_where_min_n(df):
    sol_rowindex = np.array([[[3, 1, 0, 4, -1, -1], [13, 11, 10, 12, 14, -1]], [[9, 7, 5, 6, 8, -1], [19, 17, 15, 16, 18, -1]]])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds.min_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.min('plusminus'))).data)
        agg = c.points(df, 'x', 'y', ds.where(ds.min_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.min('plusminus'), 'reverse')).data)