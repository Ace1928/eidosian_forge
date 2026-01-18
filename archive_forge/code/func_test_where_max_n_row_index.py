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
def test_where_max_n_row_index(df):
    sol = np.array([[[4, -3, nan, -1, 0, nan], [14, -13, 12, -11, 10, nan]], [[-9, 8, -7, 6, -5, nan], [-19, 18, -17, 16, -15, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.where(ds._max_n_row_index(n=n), 'plusminus'))
        out = sol[:, :, :n]
        print(n, agg.data.tolist())
        print(' ', out.tolist())
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.where(ds._max_row_index(), 'plusminus')).data)