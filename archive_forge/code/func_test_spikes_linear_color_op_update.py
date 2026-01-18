import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Spikes
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_spikes_linear_color_op_update(self):
    spikes = HoloMap({0: Spikes([(0, 0, 0.5), (0, 1, 3.2), (0, 2, 1.8)], vdims=['y', 'color']), 1: Spikes([(0, 0, 0.1), (0, 1, 0.8), (0, 2, 0.3)], vdims=['y', 'color'])}).opts(color='color', framewise=True)
    plot = mpl_renderer.get_plot(spikes)
    artist = plot.handles['artist']
    self.assertEqual(np.asarray(artist.get_array()), np.array([0.5, 3.2, 1.8]))
    self.assertEqual(artist.get_clim(), (0.5, 3.2))
    plot.update((1,))
    self.assertEqual(np.asarray(artist.get_array()), np.array([0.1, 0.8, 0.3]))
    self.assertEqual(artist.get_clim(), (0.1, 0.8))