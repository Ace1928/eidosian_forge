import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import VectorField
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_vectorfield_line_width_op_update(self):
    vectorfield = HoloMap({0: VectorField([(0, 0, 0, 1, 1), (0, 1, 0, 1, 4), (0, 2, 0, 1, 8)], vdims=['A', 'M', 'line_width']), 1: VectorField([(0, 0, 0, 1, 3), (0, 1, 0, 1, 2), (0, 2, 0, 1, 5)], vdims=['A', 'M', 'line_width'])}).opts(linewidth='line_width')
    plot = mpl_renderer.get_plot(vectorfield)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_linewidths(), [1, 4, 8])
    plot.update((1,))
    self.assertEqual(artist.get_linewidths(), [3, 2, 5])