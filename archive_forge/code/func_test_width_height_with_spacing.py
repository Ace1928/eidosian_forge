import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
def test_width_height_with_spacing(self):
    fig = figure_grid([[{'data': [{'type': 'scatter', 'y': [1, 3, 2]}], 'layout': {'width': 380, 'height': 380}}, {'data': [{'type': 'bar', 'y': [2, 3, 1]}], 'layout': {'width': 380, 'height': 380}}], [{'data': [{'type': 'scatterpolar', 'theta': [0, 90], 'r': [0.5, 1.0]}], 'layout': {'width': 380, 'height': 1140}}, {'data': [{'type': 'table', 'header': {'values': [['One', 'Two']]}}], 'layout': {'width': 380, 'height': 1140}}]], column_spacing=40, row_spacing=80, width=400, height=800)
    expected_x_domains = [[0, 0.45], [0.55, 1]]
    expected_y_domains = [[0, 0.225], [0.325, 1]]
    self.assertEqual(fig['layout']['xaxis']['domain'], expected_x_domains[0])
    self.assertEqual(fig['layout']['yaxis']['domain'], expected_y_domains[0])
    self.assertEqual(fig['layout']['xaxis2']['domain'], expected_x_domains[1])
    self.assertEqual(fig['layout']['yaxis2']['domain'], expected_y_domains[0])
    self.assertEqual(fig['layout']['polar']['domain']['x'], expected_x_domains[0])
    self.assertEqual(fig['layout']['polar']['domain']['y'], expected_y_domains[1])
    self.assertEqual(fig['data'][3]['domain']['x'], expected_x_domains[1])
    self.assertEqual(fig['data'][3]['domain']['y'], expected_y_domains[1])
    self.assertEqual(fig['layout']['width'], 400)
    self.assertEqual(fig['layout']['height'], 800)