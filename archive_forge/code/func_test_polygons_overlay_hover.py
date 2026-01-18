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
def test_polygons_overlay_hover(self):
    obj = NdOverlay({i: Polygons([{('x', 'y'): np.random.rand(10, 2), 'z': 0}], vdims=['z']) for i in range(5)}, kdims=['Test'])
    opts = {'Polygons': {'tools': ['hover']}, 'NdOverlay': {'legend_limit': 0}}
    obj = obj.opts(opts)
    self._test_hover_info(obj, [('Test', '@{Test}'), ('z', '@{z}')])