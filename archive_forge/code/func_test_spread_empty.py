import numpy as np
from holoviews.core.spaces import DynamicMap
from holoviews.element import Spread
from holoviews.streams import Buffer
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spread_empty(self):
    spread = Spread([])
    plot = bokeh_renderer.get_plot(spread)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['x'], [])
    self.assertEqual(cds.data['y'], [])