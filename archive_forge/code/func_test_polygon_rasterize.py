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
@spatialpandas_skip
def test_polygon_rasterize(self):
    poly = Polygons([{'x': [0, 1, 2], 'y': [0, 1, 0], 'holes': [[[(1.6, 0.2), (1, 0.8), (0.4, 0.2)]]]}])
    agg = rasterize(poly, width=6, height=6, dynamic=False)
    xs = [0.166667, 0.5, 0.833333, 1.166667, 1.5, 1.833333]
    ys = [0.083333, 0.25, 0.416667, 0.583333, 0.75, 0.916667]
    arr = np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]])
    expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
    self.assertEqual(agg, expected)