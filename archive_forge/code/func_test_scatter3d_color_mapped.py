import numpy as np
from holoviews.element import Scatter3D
from .test_plot import TestPlotlyPlot
def test_scatter3d_color_mapped(self):
    scatter = Scatter3D(([0, 1], [2, 3], [4, 5])).opts(color='y')
    state = self._get_plot_state(scatter)
    self.assertEqual(state['data'][0]['marker']['color'], np.array([2, 3]))
    self.assertEqual(state['data'][0]['marker']['cmin'], 2)
    self.assertEqual(state['data'][0]['marker']['cmax'], 3)