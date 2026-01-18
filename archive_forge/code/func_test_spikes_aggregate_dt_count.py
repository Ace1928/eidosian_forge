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
def test_spikes_aggregate_dt_count(self):
    spikes = Spikes([dt.datetime(2016, 1, 1), dt.datetime(2016, 1, 2), dt.datetime(2016, 1, 3)])
    agg = rasterize(spikes, width=5, dynamic=False, expand=False)
    bounds = (np.datetime64('2016-01-01T00:00:00.000000'), 0, np.datetime64('2016-01-03T00:00:00.000000'), 0.5)
    expected = Image(np.array([[1, 0, 1, 0, 1]]), vdims=Dimension('Count', nodata=0), bounds=bounds)
    self.assertEqual(agg, expected)