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
def test_where_first_n(df):
    sol_rowindex = np.array([[[0, 1, 3, 4, -1, -1], [10, 11, 12, 13, 14, -1]], [[5, 6, 7, 8, 9, -1], [15, 16, 17, 18, 19, -1]]])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds.first_n('plusminus', n=n)))
        out = sol_rowindex[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.first('plusminus'))).data)
        agg = c.points(df, 'x', 'y', ds.where(ds.first_n('plusminus', n=n), 'reverse'))
        out = sol_reverse[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds.first('plusminus'), 'reverse')).data)