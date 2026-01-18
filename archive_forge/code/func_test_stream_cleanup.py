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
def test_stream_cleanup(self):
    stream = Stream.define('Test', test=1)()
    dmap = DynamicMap(lambda test: Curve([]), streams=[stream])
    plot = bokeh_renderer.get_plot(dmap)
    self.assertTrue(bool(stream._subscribers))
    plot.cleanup()
    self.assertFalse(bool(stream._subscribers))