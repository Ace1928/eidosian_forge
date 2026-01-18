import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_plot_simple_graph(self):
    plot = bokeh_renderer.get_plot(self.graph)
    node_source = plot.handles['scatter_1_source']
    edge_source = plot.handles['multi_line_1_source']
    layout_source = plot.handles['layout_source']
    self.assertEqual(node_source.data['index'], self.source)
    self.assertEqual(edge_source.data['start'], self.source)
    self.assertEqual(edge_source.data['end'], self.target)
    layout = {z: (x, y) for x, y, z in self.graph.nodes.array()}
    self.assertEqual(layout_source.graph_layout, layout)