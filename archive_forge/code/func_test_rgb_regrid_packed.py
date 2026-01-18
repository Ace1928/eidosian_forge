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
def test_rgb_regrid_packed(self):
    coords = {'x': [1, 2], 'y': [1, 2], 'band': [0, 1, 2]}
    arr = np.array([[[255, 10], [0, 30]], [[1, 0], [0, 0]], [[127, 0], [0, 68]]]).T
    da = xr.DataArray(data=arr, dims=('x', 'y', 'band'), coords=coords)
    im = RGB(da, ['x', 'y'])
    agg = rasterize(im, width=3, height=3, dynamic=False, upsample=True)
    xs = [0.8333333, 1.5, 2.166666]
    ys = [0.8333333, 1.5, 2.166666]
    arr = np.array([[[255, 255, 10], [255, 255, 10], [0, 0, 30]], [[1, 1, 0], [1, 1, 0], [0, 0, 0]], [[127, 127, 0], [127, 127, 0], [0, 0, 68]]]).transpose((1, 2, 0))
    expected = RGB((xs, ys, arr))
    self.assertEqual(agg, expected)