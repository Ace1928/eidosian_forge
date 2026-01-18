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
def test_spikes_aggregate_count_dask(self):
    spikes = Spikes([1, 2, 3], datatype=['dask'])
    agg = rasterize(spikes, width=5, dynamic=False, expand=False)
    expected = Image(np.array([[1, 0, 1, 0, 1]]), vdims=Dimension('Count', nodata=0), xdensity=2.5, ydensity=1, bounds=(1, 0, 3, 0.5))
    self.assertEqual(agg, expected)