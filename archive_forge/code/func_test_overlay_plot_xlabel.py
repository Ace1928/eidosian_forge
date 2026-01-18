from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_overlay_plot_xlabel(self):
    overlay = Curve([]) * Curve([(10, 1), (100, 2), (1000, 3)]).opts(xlabel='X-Axis')
    state = self._get_plot_state(overlay)
    self.assertEqual(state['layout']['xaxis']['title']['text'], 'X-Axis')