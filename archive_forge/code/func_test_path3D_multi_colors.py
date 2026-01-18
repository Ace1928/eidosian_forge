import numpy as np
from holoviews.element import Path3D
from .test_plot import TestPlotlyPlot
def test_path3D_multi_colors(self):
    path3D = Path3D([[(0, 1, 0, 'red'), (1, 2, 1, 'red'), (2, 3, 2, 'red')], [(-1, 1, 3, 'blue'), (-2, 2, 4, 'blue'), (-3, 3, 5, 'blue')]], vdims='color').opts(color='color')
    state = self._get_plot_state(path3D)
    self.assertEqual(state['data'][0]['line']['color'], 'red')
    self.assertEqual(state['data'][1]['line']['color'], 'blue')