import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, FactorRange, LinearColorMapper, Scatter
from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_points_padding_nonsquare(self):
    points = Points([1, 2, 3]).opts(padding=0.1, width=600)
    plot = bokeh_renderer.get_plot(points)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, -0.1)
    self.assertEqual(x_range.end, 2.1)
    self.assertEqual(y_range.start, 0.8)
    self.assertEqual(y_range.end, 3.2)