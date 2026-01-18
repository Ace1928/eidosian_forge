import numpy as np
from holoviews.element import Scatter, Tiles
from .test_plot import TestPlotlyPlot
def test_scatter_color_mapped(self):
    scatter = Tiles('') * Scatter([3, 2, 1]).opts(color='x')
    state = self._get_plot_state(scatter)
    self.assertEqual(state['data'][1]['marker']['color'], np.array([0, 1, 2]))
    self.assertEqual(state['data'][1]['marker']['cmin'], 0)
    self.assertEqual(state['data'][1]['marker']['cmax'], 2)