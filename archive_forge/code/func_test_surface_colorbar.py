import numpy as np
from holoviews.element import Surface
from .test_plot import TestPlotlyPlot
def test_surface_colorbar(self):
    img = Surface(([1, 2, 3], [0, 1], np.array([[0, 1, 2], [2, 3, 4]])))
    img.opts(colorbar=True)
    state = self._get_plot_state(img)
    trace = state['data'][0]
    self.assertTrue(trace['showscale'])
    img.opts(colorbar=False)
    state = self._get_plot_state(img)
    trace = state['data'][0]
    self.assertFalse(trace['showscale'])