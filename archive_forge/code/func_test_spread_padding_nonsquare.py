import numpy as np
from holoviews.core.spaces import DynamicMap
from holoviews.element import Spread
from holoviews.streams import Buffer
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spread_padding_nonsquare(self):
    spread = Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1, width=600)
    plot = bokeh_renderer.get_plot(spread)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.9)
    self.assertEqual(x_range.end, 3.1)
    self.assertEqual(y_range.start, 0.19999999999999996)
    self.assertEqual(y_range.end, 3.8)