import matplotlib.pyplot as plt
import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Points
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_point_line_width_op_update(self):
    points = HoloMap({0: Points([(0, 0, 3), (0, 1, 1), (0, 2, 2)], vdims='line_width'), 1: Points([(0, 0, 2.5), (0, 1, 3), (0, 2, 1.2)], vdims='line_width')}).opts(linewidth='line_width')
    plot = mpl_renderer.get_plot(points)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_linewidths(), [3, 1, 2])
    plot.update((1,))
    self.assertEqual(artist.get_linewidths(), [2.5, 3, 1.2])