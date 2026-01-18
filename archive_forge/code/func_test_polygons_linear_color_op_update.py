import numpy as np
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import Contours, Path, Polygons
from .test_plot import TestMPLPlot, mpl_renderer
def test_polygons_linear_color_op_update(self):
    polygons = HoloMap({0: Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}], vdims='color'), 1: Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 2}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 5}], vdims='color')}).opts(color='color', framewise=True)
    plot = mpl_renderer.get_plot(polygons)
    artist = plot.handles['artist']
    self.assertEqual(np.asarray(artist.get_array()), np.array([7, 3]))
    self.assertEqual(artist.get_clim(), (3, 7))
    plot.update((1,))
    self.assertEqual(np.asarray(artist.get_array()), np.array([2, 5]))
    self.assertEqual(artist.get_clim(), (2, 5))