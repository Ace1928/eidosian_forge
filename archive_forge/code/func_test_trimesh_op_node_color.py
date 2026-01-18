import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_trimesh_op_node_color(self):
    edges = [(0, 1, 2), (1, 2, 3)]
    nodes = [(-1, -1, 0, 'red'), (0, 0, 1, 'green'), (0, 1, 2, 'blue'), (1, 0, 3, 'black')]
    trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
    plot = bokeh_renderer.get_plot(trimesh)
    cds = plot.handles['scatter_1_source']
    glyph = plot.handles['scatter_1_glyph']
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color'})
    self.assertEqual(glyph.line_color, 'black')
    self.assertEqual(cds.data['node_color'], np.array(['red', 'green', 'blue', 'black']))