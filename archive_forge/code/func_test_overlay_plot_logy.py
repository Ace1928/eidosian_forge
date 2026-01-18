from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_overlay_plot_logy(self):
    curve = (Curve([(1, 1), (2, 10), (3, 100)]) * Curve([])).opts(logy=True)
    state = self._get_plot_state(curve)
    self.assertEqual(state['layout']['yaxis']['type'], 'log')