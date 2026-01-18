import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_single_hline(self):
    hline = HLine(3)
    state = self._get_plot_state(hline)
    shapes = state['layout']['shapes']
    self.assertEqual(len(shapes), 1)
    self.assert_hline(shapes[0], 3)