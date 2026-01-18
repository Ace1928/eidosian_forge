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
def test_where_min_n_row_index(df):
    sol = np.array([[[0, -1, nan, -3, 4, nan], [10, -11, 12, -13, 14, nan]], [[-5, 6, -7, 8, -9, nan], [-15, 16, -17, 18, -19, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds._min_n_row_index(n=n), 'plusminus'))
        out = sol[:, :, :n]
        print(n, agg.data.tolist())
        print(' ', out.tolist())
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds._min_row_index(), 'plusminus')).data)