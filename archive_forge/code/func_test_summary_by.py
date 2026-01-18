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
def test_summary_by(df):
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by('cat')))
    agg_by = c.points(df, 'x', 'y', ds.by('cat'))
    assert_eq_xr(agg_summary['by'], agg_by)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by('cat'), max=ds.max('plusminus')))
    agg_max = c.points(df, 'x', 'y', ds.max('plusminus'))
    assert_eq_xr(agg_summary['by'], agg_by)
    assert_eq_xr(agg_summary['max'], agg_max)
    agg_summary = c.points(df, 'x', 'y', ds.summary(max=ds.max('plusminus'), by=ds.by('cat')))
    assert_eq_xr(agg_summary['by'], agg_by)
    assert_eq_xr(agg_summary['max'], agg_max)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by('cat'), by_any=ds.by('cat', ds.any())))
    agg_by_any = c.points(df, 'x', 'y', ds.by('cat', ds.any()))
    assert_eq_xr(agg_summary['by'], agg_by)
    assert_eq_xr(agg_summary['by_any'], agg_by_any)
    agg_summary = c.points(df, 'x', 'y', ds.summary(by=ds.by('cat'), by2=ds.by('cat2')))
    agg_by2 = c.points(df, 'x', 'y', ds.by('cat2'))
    assert_eq_xr(agg_summary['by'], agg_by)
    assert_eq_xr(agg_summary['by2'], agg_by2)