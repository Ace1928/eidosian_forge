import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_single_vline(self):
    vline = VLine(3)
    state = self._get_plot_state(vline)
    shapes = state['layout']['shapes']
    self.assertEqual(len(shapes), 1)
    self.assert_vline(shapes[0], 3)