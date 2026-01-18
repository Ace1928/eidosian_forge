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
@pytest.mark.parametrize('sel_fn', (ds.first, ds.last, ds.min, ds.max))
def test_rasterize_selector(point_plot, sel_fn):
    rast_input = dict(dynamic=False, x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img = rasterize(point_plot, selector=sel_fn('val'), **rast_input)
    assert list(img.data) == ['Count', 'index', 's', 'val', 'cat']
    assert list(img.vdims) == ['Count', 's', 'val', 'cat']
    img_agg = rasterize(point_plot, aggregator=ds.where(sel_fn('val')), **rast_input)
    for c in ['s', 'val', 'cat']:
        np.testing.assert_array_equal(img[c], img_agg[c])
    img_count = rasterize(point_plot, **rast_input)
    np.testing.assert_array_equal(img['Count'], img_count['Count'])