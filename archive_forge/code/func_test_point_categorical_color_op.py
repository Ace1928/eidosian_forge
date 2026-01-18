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
def test_point_categorical_color_op(self):
    points = Points([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')], vdims='color').opts(color='color')
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_color_mapper']
    self.assertTrue(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
    self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C']))
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})