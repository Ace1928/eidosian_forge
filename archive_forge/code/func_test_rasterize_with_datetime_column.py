import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
def test_rasterize_with_datetime_column():
    n = 4
    df = pd.DataFrame({'x': np.random.uniform(-180, 180, n), 'y': np.random.uniform(-90, 90, n), 'Timestamp': pd.date_range(start='2023-01-01', periods=n, freq='D'), 'Value': np.random.rand(n) * 100})
    point_plot = Points(df)
    rast_input = dict(dynamic=False, x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img_agg = rasterize(point_plot, selector=ds.first('Value'), **rast_input)
    assert img_agg['Timestamp'].dtype == np.dtype('datetime64[ns]')