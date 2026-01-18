import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Spikes
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_spikes_color_op_update(self):
    spikes = HoloMap({0: Spikes([(0, 0, '#000000'), (0, 1, '#FF0000'), (0, 2, '#00FF00')], vdims=['y', 'color']), 1: Spikes([(0, 0, '#FF0000'), (0, 1, '#00FF00'), (0, 2, '#0000FF')], vdims=['y', 'color'])}).opts(color='color')
    plot = mpl_renderer.get_plot(spikes)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_edgecolors(), np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]))
    plot.update((1,))
    self.assertEqual(artist.get_edgecolors(), np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]))