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
def test_rasterize_summerize(point_plot):
    agg_fn_count, agg_fn_first = (ds.count(), ds.first('val'))
    agg_fn = ds.summary(count=agg_fn_count, first=agg_fn_first)
    rast_input = dict(dynamic=False, x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img_sum = rasterize(point_plot, aggregator=agg_fn, **rast_input)
    img_count = rasterize(point_plot, aggregator=agg_fn_count, **rast_input)
    img_first = rasterize(point_plot, aggregator=agg_fn_first, **rast_input)
    np.testing.assert_array_equal(img_sum['first'], img_first['val'])
    np.testing.assert_array_equal(img_sum['count'], np.nan_to_num(img_count['Count']))