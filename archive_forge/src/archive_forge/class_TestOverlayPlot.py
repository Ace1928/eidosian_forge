from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
class TestOverlayPlot(TestPlotlyPlot):

    def test_overlay_state(self):
        layout = Curve([1, 2, 3]) * Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['layout']['yaxis']['range'], [1, 6])

    def test_overlay_plot_logx(self):
        curve = (Curve([(10, 1), (100, 2), (1000, 3)]) * Curve([])).opts(logx=True)
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['xaxis']['type'], 'log')

    def test_overlay_plot_logy(self):
        curve = (Curve([(1, 1), (2, 10), (3, 100)]) * Curve([])).opts(logy=True)
        state = self._get_plot_state(curve)
        self.assertEqual(state['layout']['yaxis']['type'], 'log')

    def test_overlay_plot_logz(self):
        scatter = (Scatter3D([(0, 1, 10), (1, 2, 100), (2, 3, 1000)]) * Path3D([])).opts(logz=True)
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['type'], 'log')

    def test_overlay_plot_xlabel(self):
        overlay = Curve([]) * Curve([(10, 1), (100, 2), (1000, 3)]).opts(xlabel='X-Axis')
        state = self._get_plot_state(overlay)
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'X-Axis')

    def test_overlay_plot_ylabel(self):
        overlay = Curve([]) * Curve([(10, 1), (100, 2), (1000, 3)]).opts(ylabel='Y-Axis')
        state = self._get_plot_state(overlay)
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'Y-Axis')

    def test_overlay_plot_zlabel(self):
        scatter = Path3D([]) * Scatter3D([(10, 1, 2), (100, 2, 3), (1000, 3, 5)]).opts(zlabel='Z-Axis')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['layout']['scene']['zaxis']['title']['text'], 'Z-Axis')