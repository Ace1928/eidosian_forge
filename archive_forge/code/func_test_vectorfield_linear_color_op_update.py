import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import VectorField
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_vectorfield_linear_color_op_update(self):
    vectorfield = HoloMap({0: VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 1), (0, 2, 0, 1, 2)], vdims=['A', 'M', 'color']), 1: VectorField([(0, 0, 0, 1, 3.2), (0, 1, 0, 1, 2), (0, 2, 0, 1, 4)], vdims=['A', 'M', 'color'])}).opts(color='color', framewise=True)
    plot = mpl_renderer.get_plot(vectorfield)
    artist = plot.handles['artist']
    self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 2]))
    self.assertEqual(artist.get_clim(), (0, 2))
    plot.update((1,))
    self.assertEqual(np.asarray(artist.get_array()), np.array([3.2, 2, 4]))
    self.assertEqual(artist.get_clim(), (2, 4))