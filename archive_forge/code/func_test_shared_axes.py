import datetime as dt
import re
import numpy as np
from bokeh.models import Div, GlyphRenderer, GridPlot, Spacer, Tabs, Title, Toolbar
from bokeh.models.layouts import TabPanel
from bokeh.plotting import figure
from holoviews.core import (
from holoviews.element import Curve, Histogram, Image, Points, Scatter
from holoviews.streams import Stream
from holoviews.util import opts, render
from holoviews.util.transform import dim
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_shared_axes(self):
    curve = Curve(range(10))
    img = Image(np.random.rand(10, 10))
    plot = bokeh_renderer.get_plot(curve + img)
    plot = plot.subplots[0, 1].subplots['main']
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual((x_range.start, x_range.end), (-0.5, 9))
    self.assertEqual((y_range.start, y_range.end), (-0.5, 9))