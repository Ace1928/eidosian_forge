from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_radial_heatmap_ranges(self):
    hm = HeatMap([(0, 0, 1), (0, 1, 2), (1, 0, 3)]).opts(radial=True, colorbar=True)
    plot = bokeh_renderer.get_plot(hm)
    self.assertEqual(plot.handles['x_range'].start, -0.05)
    self.assertEqual(plot.handles['x_range'].end, 1.05)
    self.assertEqual(plot.handles['y_range'].start, -0.05)
    self.assertEqual(plot.handles['y_range'].end, 1.05)