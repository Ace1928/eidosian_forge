import numpy as np
from holoviews.element import Violin
from .test_plot import TestPlotlyPlot
def test_violin_multi(self):
    violin = Violin((['A'] * 8 + ['B'] * 8, [1, 1, 2, 3, 3, 4, 5, 5] * 2), 'x', 'y')
    state = self._get_plot_state(violin)
    self.assertEqual(len(state['data']), 2)
    self.assertEqual(state['data'][0]['type'], 'violin')
    self.assertEqual(state['data'][0]['name'], 'A')
    self.assertEqual(state['data'][0]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
    self.assertEqual(state['data'][1]['type'], 'violin')
    self.assertEqual(state['data'][1]['name'], 'B')
    self.assertEqual(state['data'][1]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
    self.assertEqual(state['layout']['xaxis']['title']['text'], 'x')
    self.assertEqual(state['layout']['yaxis']['range'], [1, 5])
    self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')