import matplotlib.pyplot as plt
import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Points
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_points_sizes_scalar_update(self):
    hmap = HoloMap({i: Points([1, 2, 3]).opts(s=i * 10) for i in range(1, 3)})
    plot = mpl_renderer.get_plot(hmap)
    artist = plot.handles['artist']
    plot.update((1,))
    self.assertEqual(artist.get_sizes(), np.array([10]))
    plot.update((2,))
    self.assertEqual(artist.get_sizes(), np.array([20]))