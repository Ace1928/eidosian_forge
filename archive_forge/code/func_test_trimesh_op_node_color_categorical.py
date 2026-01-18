import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_trimesh_op_node_color_categorical(self):
    edges = [(0, 1, 2), (1, 2, 3)]
    nodes = [(-1, -1, 0, 'B'), (0, 0, 1, 'C'), (0, 1, 2, 'A'), (1, 0, 3, 'B')]
    trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
    plot = bokeh_renderer.get_plot(trimesh)
    cds = plot.handles['scatter_1_source']
    glyph = plot.handles['scatter_1_glyph']
    cmapper = plot.handles['node_color_color_mapper']
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color', 'transform': cmapper})
    self.assertEqual(glyph.line_color, 'black')
    self.assertEqual(cds.data['node_color'], np.array(['B', 'C', 'A', 'B']))