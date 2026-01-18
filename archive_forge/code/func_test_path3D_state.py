import numpy as np
from holoviews.element import Path3D
from .test_plot import TestPlotlyPlot
def test_path3D_state(self):
    path3D = Path3D([(0, 1, 0), (1, 2, 1), (2, 3, 2)])
    state = self._get_plot_state(path3D)
    self.assertEqual(state['data'][0]['x'], np.array([0, 1, 2]))
    self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
    self.assertEqual(state['data'][0]['mode'], 'lines')
    self.assertEqual(state['data'][0]['type'], 'scatter3d')
    self.assertEqual(state['layout']['scene']['xaxis']['range'], [0, 2])
    self.assertEqual(state['layout']['scene']['yaxis']['range'], [1, 3])
    self.assertEqual(state['layout']['scene']['zaxis']['range'], [0, 2])