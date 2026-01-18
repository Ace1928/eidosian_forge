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
def test_rectangles_aggregate_dt_count(self):
    rects = Rectangles([(0, dt.datetime(2016, 1, 2), 4, dt.datetime(2016, 1, 3)), (1, dt.datetime(2016, 1, 1), 2, dt.datetime(2016, 1, 5))])
    agg = rasterize(rects, width=4, height=4, dynamic=False)
    xs = [0.5, 1.5, 2.5, 3.5]
    ys = [np.datetime64('2016-01-01T12:00:00'), np.datetime64('2016-01-02T12:00:00'), np.datetime64('2016-01-03T12:00:00'), np.datetime64('2016-01-04T12:00:00')]
    arr = np.array([[0, 1, 1, 0], [1, 2, 2, 1], [0, 1, 1, 0], [0, 0, 0, 0]])
    bounds = (0.0, np.datetime64('2016-01-01T00:00:00'), 4.0, np.datetime64('2016-01-05T00:00:00'))
    expected = Image((xs, ys, arr), bounds=bounds, vdims=Dimension('Count', nodata=0))
    self.assertEqual(agg, expected)