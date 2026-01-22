import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
class TestBokehTriMeshPlot(TestBokehPlot):

    def setUp(self):
        super().setUp()
        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1.0, 0, 2), (1.5, 1, 3)]
        self.simplices = [(0, 1, 2, 0), (1, 2, 3, 1)]
        self.trimesh = TriMesh((self.simplices, self.nodes))
        self.trimesh_weighted = TriMesh((self.simplices, self.nodes), vdims='weight')

    def test_plot_simple_trimesh(self):
        plot = bokeh_renderer.get_plot(self.trimesh)
        node_source = plot.handles['scatter_1_source']
        edge_source = plot.handles['multi_line_1_source']
        layout_source = plot.handles['layout_source']
        self.assertEqual(node_source.data['index'], np.arange(4))
        self.assertEqual(edge_source.data['start'], np.arange(2))
        self.assertEqual(edge_source.data['end'], np.arange(1, 3))
        layout = {z: (x, y) for x, y, z in self.trimesh.nodes.array()}
        self.assertEqual(layout_source.graph_layout, layout)

    def test_plot_simple_trimesh_filled(self):
        plot = bokeh_renderer.get_plot(self.trimesh.opts(filled=True))
        node_source = plot.handles['scatter_1_source']
        edge_source = plot.handles['patches_1_source']
        layout_source = plot.handles['layout_source']
        self.assertIsInstance(plot.handles['patches_1_glyph'], Patches)
        self.assertEqual(node_source.data['index'], np.arange(4))
        self.assertEqual(edge_source.data['start'], np.arange(2))
        self.assertEqual(edge_source.data['end'], np.arange(1, 3))
        layout = {z: (x, y) for x, y, z in self.trimesh.nodes.array()}
        self.assertEqual(layout_source.graph_layout, layout)

    def test_trimesh_edges_categorical_colormapped(self):
        g = self.trimesh.opts(edge_color_index='node1', edge_cmap=['#FFFFFF', '#000000'])
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_colormapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        factors = ['0', '1', '2', '3']
        self.assertEqual(cmapper.factors, factors)
        self.assertEqual(edge_source.data['node1_str__'], ['0', '1'])
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'node1_str__', 'transform': cmapper})

    def test_trimesh_nodes_numerically_colormapped(self):
        g = self.trimesh_weighted.opts(edge_color_index='weight', edge_cmap=['#FFFFFF', '#000000'])
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_colormapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 1)
        self.assertEqual(edge_source.data['weight'], np.array([0, 1]))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'weight', 'transform': cmapper})

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

    def test_trimesh_op_node_color_linear(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 2), (0, 0, 1, 1), (0, 1, 2, 3), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        cmapper = plot.handles['node_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['node_color'], np.array([2, 1, 3, 4]))
        self.assertEqual(cmapper.low, 1)
        self.assertEqual(cmapper.high, 4)

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

    def test_trimesh_op_node_size(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 3), (0, 0, 1, 2), (0, 1, 2, 8), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='size'))).opts(node_size='size')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.size), {'field': 'node_size'})
        self.assertEqual(cds.data['node_size'], np.array([3, 2, 8, 4]))

    def test_trimesh_op_node_alpha(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 0.2), (0, 0, 1, 0.6), (0, 1, 2, 1), (1, 0, 3, 0.3)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='alpha'))).opts(node_alpha='alpha')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'node_alpha'})
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'node_alpha'})
        self.assertEqual(cds.data['node_alpha'], np.array([0.2, 0.6, 1, 0.3]))

    def test_trimesh_op_node_line_width(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 0.2), (0, 0, 1, 0.6), (0, 1, 2, 1), (1, 0, 3, 0.3)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='line_width'))).opts(node_line_width='line_width')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'node_line_width'})
        self.assertEqual(cds.data['node_line_width'], np.array([0.2, 0.6, 1, 0.3]))

    def test_trimesh_op_edge_color_linear_mean_node(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 2), (0, 0, 1, 1), (0, 1, 2, 3), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(edge_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(cds.data['edge_color'], np.array([2, 8 / 3.0]))
        self.assertEqual(cmapper.low, 1)
        self.assertEqual(cmapper.high, 4)

    def test_trimesh_op_edge_color(self):
        edges = [(0, 1, 2, 'red'), (1, 2, 3, 'blue')]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color'})
        self.assertEqual(cds.data['edge_color'], np.array(['red', 'blue']))

    def test_trimesh_op_edge_color_linear(self):
        edges = [(0, 1, 2, 2.4), (1, 2, 3, 3.6)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(cds.data['edge_color'], np.array([2.4, 3.6]))
        self.assertEqual(cmapper.low, 2.4)
        self.assertEqual(cmapper.high, 3.6)

    def test_trimesh_op_edge_color_linear_filled(self):
        edges = [(0, 1, 2, 2.4), (1, 2, 3, 3.6)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color', filled=True)
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['patches_1_source']
        glyph = plot.handles['patches_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['edge_color'], np.array([2.4, 3.6]))
        self.assertEqual(cmapper.low, 2.4)
        self.assertEqual(cmapper.high, 3.6)

    def test_trimesh_op_edge_color_categorical(self):
        edges = [(0, 1, 2, 'A'), (1, 2, 3, 'B')]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(cds.data['edge_color'], np.array(['A', 'B']))
        self.assertEqual(cmapper.factors, ['A', 'B'])

    def test_trimesh_op_edge_alpha(self):
        edges = [(0, 1, 2, 0.7), (1, 2, 3, 0.3)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='alpha').opts(edge_alpha='alpha')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'edge_alpha'})
        self.assertEqual(cds.data['edge_alpha'], np.array([0.7, 0.3]))

    def test_trimesh_op_edge_line_width(self):
        edges = [(0, 1, 2, 7), (1, 2, 3, 3)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='line_width').opts(edge_line_width='line_width')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'edge_line_width'})
        self.assertEqual(cds.data['edge_line_width'], np.array([7, 3]))