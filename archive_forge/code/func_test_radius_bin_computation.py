from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_radius_bin_computation(self):
    """Test computation of bins for radius/annulars.

        """
    order = sorted(self.ann_bins.keys())
    values = self.plot._get_bins('radius', order)
    self.assertEqual(values.keys(), self.ann_bins.keys())
    self.assertEqual(values['o1'], self.ann_bins['o1'])
    self.assertEqual(values['o2'], self.ann_bins['o2'])
    values = self.plot._get_bins('radius', order, reverse=True)
    self.assertEqual(values.keys(), self.ann_bins.keys())
    self.assertEqual(values['o1'], self.ann_bins['o2'])
    self.assertEqual(values['o2'], self.ann_bins['o1'])