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
def test_rasterize_image_expand_default(self):
    assert not regrid.expand
    data = np.arange(100.0).reshape(10, 10)
    c = np.arange(10.0)
    da = xr.DataArray(data, coords=dict(x=c, y=c))
    rast_input = dict(x_range=(-1, 10), y_range=(-1, 10), precompute=True, dynamic=False)
    img = rasterize(Image(da), **rast_input)
    output = img.data['z'].to_numpy()
    np.testing.assert_array_equal(output, data.T)
    assert not np.isnan(output).any()
    img = rasterize(Image(da), expand=True, **rast_input)
    output = img.data['z'].to_numpy()
    assert np.isnan(output).any()