import matplotlib.pyplot as plt
import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Points
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_point_color_op_update(self):
    points = HoloMap({0: Points([(0, 0, '#000000'), (0, 1, '#FF0000'), (0, 2, '#00FF00')], vdims='color'), 1: Points([(0, 0, '#0000FF'), (0, 1, '#00FF00'), (0, 2, '#FF0000')], vdims='color')}).opts(color='color')
    plot = mpl_renderer.get_plot(points)
    artist = plot.handles['artist']
    plot.update((1,))
    self.assertEqual(artist.get_facecolors(), np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]]))