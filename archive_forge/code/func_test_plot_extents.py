from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_plot_extents(self):
    """Test correct computation of extents.

        """
    extents = self.plot.get_extents('', '')
    self.assertEqual(extents, (-0.2, -0.2, 2.2, 2.2))