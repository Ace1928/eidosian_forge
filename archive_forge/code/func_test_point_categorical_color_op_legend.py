import matplotlib.pyplot as plt
import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Points
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_point_categorical_color_op_legend(self):
    points = Points([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'A')], vdims='color').opts(color='color', show_legend=True)
    plot = mpl_renderer.get_plot(points)
    leg = plot.handles['axis'].get_legend()
    legend_labels = [l.get_text() for l in leg.texts]
    self.assertEqual(legend_labels, ['A', 'B'])