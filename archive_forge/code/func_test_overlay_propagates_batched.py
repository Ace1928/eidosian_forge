import datetime as dt
from unittest import SkipTest
import numpy as np
import panel as pn
import pytest
from bokeh.document import Document
from bokeh.models import (
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.util import dt_to_int
from holoviews.element import Curve, HeatMap, Image, Labels, Scatter
from holoviews.plotting.util import process_cmap
from holoviews.streams import PointDraw, Stream
from holoviews.util import render
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_overlay_propagates_batched(self):
    overlay = NdOverlay({i: Curve([1, 2, 3]).opts(yformatter='%.1f') for i in range(10)}).opts(yformatter='%.3f', legend_limit=1)
    plot = bokeh_renderer.get_plot(overlay)
    self.assertEqual(plot.state.yaxis.formatter.format, '%.3f')