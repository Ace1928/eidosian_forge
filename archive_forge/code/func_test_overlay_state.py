from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_overlay_state(self):
    layout = Curve([1, 2, 3]) * Curve([2, 4, 6])
    state = self._get_plot_state(layout)
    self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
    self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
    self.assertEqual(state['layout']['yaxis']['range'], [1, 6])