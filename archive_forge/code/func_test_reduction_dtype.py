from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('reduction,dtype,aa_dtype', [(ds.any(), bool, np.float32), (ds.count(), np.uint32, np.float32), (ds.max('value'), np.float64, np.float64), (ds.min('value'), np.float64, np.float64), (ds.sum('value'), np.float64, np.float64), (ds.where(ds.max('value')), np.int64, np.int64), (ds.where(ds.max('value'), 'other'), np.float64, np.float64)])
def test_reduction_dtype(reduction, dtype, aa_dtype):
    cvs = ds.Canvas(plot_width=10, plot_height=10)
    df = pd.DataFrame(dict(x=[0, 1], y=[1, 2], value=[1, 2], other=[1.2, 3.4]))
    agg = cvs.line(df, 'x', 'y', line_width=0, agg=reduction)
    assert agg.dtype == dtype
    if not isinstance(reduction, ds.where):
        agg = cvs.line(df, 'x', 'y', line_width=1, agg=reduction)
        assert agg.dtype == aa_dtype