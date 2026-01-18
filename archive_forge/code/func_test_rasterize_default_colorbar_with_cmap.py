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
def test_rasterize_default_colorbar_with_cmap(self):
    cmap = 'Reds'
    plot = self.df.hvplot.scatter('x', 'y', dynamic=False, rasterize=True, cmap=cmap)
    opts = Store.lookup_options('bokeh', plot, 'style').kwargs
    self.assertEqual(opts.get('cmap'), cmap)
    opts = Store.lookup_options('bokeh', plot, 'plot').kwargs
    self.assertTrue(opts.get('colorbar'))