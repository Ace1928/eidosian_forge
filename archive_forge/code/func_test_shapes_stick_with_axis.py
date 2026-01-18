import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
def test_shapes_stick_with_axis(self):
    fig = figure_grid([[{'data': [{'type': 'scatter', 'y': [1, 3, 2]}], 'layout': {'width': 400, 'height': 400, 'shapes': [{'type': 'rect', 'xref': 'x', 'yref': 'y', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1}, {'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': 1, 'y0': 1, 'x1': 2, 'y1': 2}]}}, {'data': [{'type': 'bar', 'y': [2, 3, 1]}], 'layout': {'width': 400, 'height': 400, 'shapes': [{'type': 'line', 'xref': 'x', 'yref': 'y', 'x0': 2, 'y0': 0, 'x1': 1, 'y1': 3}, {'type': 'path', 'xref': 'x', 'yref': 'y', 'x0': 3, 'y0': 0, 'x1': 3, 'y1': 6}]}}]])
    go.Figure(fig)
    shapes = fig['layout']['shapes']
    self.assertEqual(len(shapes), 4)
    self.assertEqual(shapes[0]['type'], 'rect')
    self.assertEqual(shapes[0]['xref'], 'x')
    self.assertEqual(shapes[0]['yref'], 'y')
    self.assertEqual(shapes[1]['type'], 'circle')
    self.assertEqual(shapes[1]['xref'], 'x')
    self.assertEqual(shapes[1]['yref'], 'y')
    self.assertEqual(shapes[2]['type'], 'line')
    self.assertEqual(shapes[2]['xref'], 'x2')
    self.assertEqual(shapes[2]['yref'], 'y2')
    self.assertEqual(shapes[3]['type'], 'path')
    self.assertEqual(shapes[3]['xref'], 'x2')
    self.assertEqual(shapes[3]['yref'], 'y2')