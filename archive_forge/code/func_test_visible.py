import numpy as np
from holoviews.element import QuadMesh
from .test_plot import TestPlotlyPlot
def test_visible(self):
    element = QuadMesh(([1, 2, 4], [0, 1], np.array([[0, 1, 2], [2, 3, 4]]))).opts(visible=False)
    state = self._get_plot_state(element)
    self.assertEqual(state['data'][0]['visible'], False)