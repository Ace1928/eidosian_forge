import numpy as np
from holoviews.element import Violin
from .test_plot import TestPlotlyPlot
def test_violin_single_invert_axes(self):
    violin = Violin([1, 1, 2, 3, 3, 4, 5, 5]).opts(invert_axes=True)
    state = self._get_plot_state(violin)
    self.assertEqual(len(state['data']), 1)
    self.assertEqual(state['data'][0]['type'], 'violin')
    self.assertEqual(state['data'][0]['name'], '')
    self.assertEqual(state['data'][0]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
    self.assertEqual(state['layout'].get('yaxis', {}), {})
    self.assertEqual(state['layout']['xaxis']['range'], [1, 5])
    self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')