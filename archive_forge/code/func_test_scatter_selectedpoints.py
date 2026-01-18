import numpy as np
from holoviews.element import Scatter, Tiles
from .test_plot import TestPlotlyPlot
def test_scatter_selectedpoints(self):
    scatter = Tiles('') * Scatter([(0, 1), (1, 2), (2, 3)]).opts(selectedpoints=[1, 2])
    state = self._get_plot_state(scatter)
    self.assertEqual(state['data'][1]['selectedpoints'], [1, 2])