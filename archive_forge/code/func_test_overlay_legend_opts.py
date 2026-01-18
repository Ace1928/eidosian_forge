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
def test_overlay_legend_opts(self):
    overlay = (Curve(np.random.randn(10).cumsum(), label='A') * Curve(np.random.randn(10).cumsum(), label='B')).opts(legend_opts={'background_fill_alpha': 0.5, 'background_fill_color': 'red'})
    plot = bokeh_renderer.get_plot(overlay)
    legend = plot.state.legend
    self.assertEqual(legend.background_fill_alpha, 0.5)
    self.assertEqual(legend.background_fill_color, 'red')