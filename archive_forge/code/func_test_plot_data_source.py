from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_plot_data_source(self):
    """Test initialization of ColumnDataSources.

        """
    source_ann = self.plot.handles['annular_wedge_1_source'].data
    self.assertEqual(list(source_ann['x']), list(self.x))
    self.assertEqual(list(source_ann['y']), list(self.y))
    self.assertEqual(list(source_ann['z']), self.z)