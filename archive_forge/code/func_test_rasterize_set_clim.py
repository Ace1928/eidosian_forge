import sys
from unittest import SkipTest
from parameterized import parameterized
import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import pytest
from holoviews import Store, render
from holoviews.element import Image, QuadMesh, Points
from holoviews.core.spaces import DynamicMap
from holoviews.core.overlay import Overlay
from holoviews.element.chart import Scatter
from holoviews.element.comparison import ComparisonTestCase
from hvplot.converter import HoloViewsConverter
from hvplot.tests.util import makeTimeDataFrame
from packaging.version import Version
def test_rasterize_set_clim(self):
    plot = self.df.hvplot.scatter('x', 'y', dynamic=False, rasterize=True, clim=(1, 4))
    opts = Store.lookup_options('bokeh', plot, 'plot').kwargs
    self.assertEqual(opts.get('clim'), (1, 4))