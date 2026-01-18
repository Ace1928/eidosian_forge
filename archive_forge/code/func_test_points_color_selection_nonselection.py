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
def test_points_color_selection_nonselection(self):
    opts = dict(color='green', selection_color='red', nonselection_color='blue')
    points = Points([(i, i * 2, i * 3, chr(65 + i)) for i in range(10)], vdims=['a', 'b']).opts(**opts)
    plot = bokeh_renderer.get_plot(points)
    glyph_renderer = plot.handles['glyph_renderer']
    self.assertEqual(glyph_renderer.glyph.fill_color, 'green')
    self.assertEqual(glyph_renderer.glyph.line_color, 'green')
    self.assertEqual(glyph_renderer.selection_glyph.fill_color, 'red')
    self.assertEqual(glyph_renderer.selection_glyph.line_color, 'red')
    self.assertEqual(glyph_renderer.nonselection_glyph.fill_color, 'blue')
    self.assertEqual(glyph_renderer.nonselection_glyph.line_color, 'blue')