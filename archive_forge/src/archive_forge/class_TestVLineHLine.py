import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
class TestVLineHLine(TestShape):

    def assert_vline(self, shape, x, xref='x', ydomain=(0, 1)):
        self.assertEqual(shape['type'], 'line')
        self.assertEqual(shape['x0'], x)
        self.assertEqual(shape['x1'], x)
        self.assertEqual(shape['xref'], xref)
        self.assertEqual(shape['y0'], ydomain[0])
        self.assertEqual(shape['y1'], ydomain[1])
        self.assertEqual(shape['yref'], 'paper')

    def assert_hline(self, shape, y, yref='y', xdomain=(0, 1)):
        self.assertEqual(shape['type'], 'line')
        self.assertEqual(shape['y0'], y)
        self.assertEqual(shape['y1'], y)
        self.assertEqual(shape['yref'], yref)
        self.assertEqual(shape['x0'], xdomain[0])
        self.assertEqual(shape['x1'], xdomain[1])
        self.assertEqual(shape['xref'], 'paper')

    def test_single_vline(self):
        vline = VLine(3)
        state = self._get_plot_state(vline)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 1)
        self.assert_vline(shapes[0], 3)

    def test_single_hline(self):
        hline = HLine(3)
        state = self._get_plot_state(hline)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 1)
        self.assert_hline(shapes[0], 3)

    def test_vline_layout(self):
        layout = (VLine(1) + VLine(2) + VLine(3) + VLine(4)).cols(2).opts(vspacing=0, hspacing=0)
        state = self._get_plot_state(layout)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 4)
        self.assert_vline(shapes[0], 3, xref='x', ydomain=[0.0, 0.5])
        self.assert_vline(shapes[1], 4, xref='x2', ydomain=[0.0, 0.5])
        self.assert_vline(shapes[2], 1, xref='x3', ydomain=[0.5, 1.0])
        self.assert_vline(shapes[3], 2, xref='x4', ydomain=[0.5, 1.0])

    def test_hline_layout(self):
        layout = (HLine(1) + HLine(2) + HLine(3) + HLine(4)).cols(2).opts(vspacing=0, hspacing=0)
        state = self._get_plot_state(layout)
        shapes = state['layout']['shapes']
        self.assertEqual(len(shapes), 4)
        self.assert_hline(shapes[0], 3, yref='y', xdomain=[0.0, 0.5])
        self.assert_hline(shapes[1], 4, yref='y2', xdomain=[0.5, 1.0])
        self.assert_hline(shapes[2], 1, yref='y3', xdomain=[0.0, 0.5])
        self.assert_hline(shapes[3], 2, yref='y4', xdomain=[0.5, 1.0])

    def test_vline_styling(self):
        self.assert_shape_element_styling(VLine(3))

    def test_hline_styling(self):
        self.assert_shape_element_styling(HLine(3))