import matplotlib.pyplot as plt
import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Points
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_points_rcparams_do_not_persist(self):
    opts = dict(fig_rcparams={'text.usetex': True})
    points = Points(([0, 1], [0, 3])).opts(**opts)
    mpl_renderer.get_plot(points)
    self.assertFalse(plt.rcParams['text.usetex'])