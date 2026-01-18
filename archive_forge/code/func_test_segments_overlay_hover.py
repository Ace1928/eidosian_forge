import pandas as pd
from bokeh.models import FactorRange
from holoviews.core import NdOverlay
from holoviews.element import Segments
from .test_plot import TestBokehPlot, bokeh_renderer
def test_segments_overlay_hover(self):
    obj = NdOverlay({i: Segments((range(31), range(31), range(1, 32), range(31))) for i in range(5)}, kdims=['Test']).opts({'Segments': {'tools': ['hover']}})
    tooltips = [('Test', '@{Test}'), ('x0', '@{x0}'), ('y0', '@{y0}'), ('x1', '@{x1}'), ('y1', '@{y1}')]
    self._test_hover_info(obj, tooltips)