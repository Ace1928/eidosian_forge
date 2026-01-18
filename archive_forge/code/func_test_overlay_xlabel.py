import numpy as np
from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve, Scatter
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_overlay_xlabel(self):
    overlay = (Curve(range(10)) * Curve(range(10))).opts(xlabel='custom x-label')
    axes = mpl_renderer.get_plot(overlay).handles['axis']
    self.assertEqual(axes.get_xlabel(), 'custom x-label')