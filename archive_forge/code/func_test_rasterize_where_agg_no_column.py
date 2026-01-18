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
@pytest.mark.parametrize('agg_input_fn,index_col', ([ds.first, [311, 433, 309, 482]], [ds.last, [491, 483, 417, 482]], [ds.min, [311, 433, 309, 482]], [ds.max, [404, 433, 417, 482]]))
def test_rasterize_where_agg_no_column(point_plot, agg_input_fn, index_col):
    agg_fn = ds.where(agg_input_fn('val'))
    rast_input = dict(dynamic=False, x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img = rasterize(point_plot, aggregator=agg_fn, **rast_input)
    assert list(img.data) == ['index', 's', 'val', 'cat']
    assert list(img.vdims) == ['val', 's', 'cat']
    np.testing.assert_array_equal(img.data['index'].data.flatten(), index_col)
    img_simple = rasterize(point_plot, aggregator=agg_input_fn('val'), **rast_input)
    np.testing.assert_array_equal(img_simple['val'], img['val'])