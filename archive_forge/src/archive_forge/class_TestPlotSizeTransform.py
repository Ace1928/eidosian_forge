from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
class TestPlotSizeTransform(ComparisonTestCase):

    def test_plotsize_initial_contents_1(self):
        plotsize = PlotSize(width=300, height=400, scale=0.5)
        self.assertEqual(plotsize.contents, {'width': 300, 'height': 400, 'scale': 0.5})

    def test_plotsize_update_1(self):
        plotsize = PlotSize(scale=0.5)
        plotsize.event(width=300, height=400)
        self.assertEqual(plotsize.contents, {'width': 150, 'height': 200, 'scale': 0.5})

    def test_plotsize_initial_contents_2(self):
        plotsize = PlotSize(width=600, height=100, scale=2)
        self.assertEqual(plotsize.contents, {'width': 600, 'height': 100, 'scale': 2})

    def test_plotsize_update_2(self):
        plotsize = PlotSize(scale=2)
        plotsize.event(width=600, height=100)
        self.assertEqual(plotsize.contents, {'width': 1200, 'height': 200, 'scale': 2})