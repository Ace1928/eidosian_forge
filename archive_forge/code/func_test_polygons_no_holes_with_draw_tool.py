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
def test_polygons_no_holes_with_draw_tool(self):
    from bokeh.models import Patches
    xs = [1, 2, 3, np.nan, 6, 7, 3]
    ys = [2, 0, 7, np.nan, 7, 5, 2]
    holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
    poly = HoloMap({0: Polygons([{'x': xs, 'y': ys, 'holes': holes}]), 1: Polygons([{'x': xs, 'y': ys}])})
    PolyDraw(source=poly)
    plot = bokeh_renderer.get_plot(poly)
    glyph = plot.handles['glyph']
    self.assertFalse(plot._has_holes)
    self.assertIsInstance(glyph, Patches)