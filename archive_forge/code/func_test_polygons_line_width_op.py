import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import HoloMap, NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Contours, Path, Polygons
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import PolyDraw
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_polygons_line_width_op(self):
    polygons = Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}], vdims='line_width').opts(line_width='line_width')
    plot = bokeh_renderer.get_plot(polygons)
    cds = plot.handles['source']
    glyph = plot.handles['glyph']
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})
    self.assertEqual(cds.data['line_width'], np.array([7, 3]))