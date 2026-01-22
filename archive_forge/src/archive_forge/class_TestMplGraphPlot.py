import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.options import AbbreviatedException, Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Chord, Graph, Nodes, TriMesh, circular_layout
from holoviews.util.transform import dim
from .test_plot import TestMPLPlot, mpl_renderer
class TestMplGraphPlot(TestMPLPlot):

    def setUp(self):
        super().setUp()
        N = 8
        self.nodes = circular_layout(np.arange(N, dtype=np.int32))
        self.source = np.arange(N, dtype=np.int32)
        self.target = np.zeros(N, dtype=np.int32)
        self.weights = np.random.rand(N)
        self.graph = Graph(((self.source, self.target),))
        self.node_info = Dataset(['Output'] + ['Input'] * (N - 1), vdims=['Label'])
        self.node_info2 = Dataset(self.weights, vdims='Weight')
        self.graph2 = Graph(((self.source, self.target), self.node_info))
        self.graph3 = Graph(((self.source, self.target), self.node_info2))
        self.graph4 = Graph(((self.source, self.target, self.weights),), vdims='Weight')

    def test_plot_simple_graph(self):
        plot = mpl_renderer.get_plot(self.graph)
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(nodes.get_offsets()), self.graph.nodes.array([0, 1]))
        self.assertEqual([p.vertices for p in edges.get_paths()], [p.array() for p in self.graph.edgepaths.split()])

    def test_plot_graph_categorical_colored_nodes(self):
        g = self.graph2.opts(color_index='Label', cmap='Set1')
        plot = mpl_renderer.get_plot(g)
        nodes = plot.handles['nodes']
        facecolors = np.array([[0.89411765, 0.10196078, 0.10980392, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0]])
        self.assertEqual(nodes.get_facecolors(), facecolors)

    def test_plot_graph_numerically_colored_nodes(self):
        g = self.graph3.opts(color_index='Weight', cmap='viridis')
        plot = mpl_renderer.get_plot(g)
        nodes = plot.handles['nodes']
        self.assertEqual(np.asarray(nodes.get_array()), self.weights)
        self.assertEqual(nodes.get_clim(), (self.weights.min(), self.weights.max()))

    def test_plot_graph_categorical_colored_edges(self):
        g = self.graph3.opts(edge_color_index='start', edge_cmap=['#FFFFFF', '#000000'])
        plot = mpl_renderer.get_plot(g)
        edges = plot.handles['edges']
        colors = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(edges.get_colors(), colors)

    def test_plot_graph_numerically_colored_edges(self):
        g = self.graph4.opts(edge_color_index='Weight', edge_cmap=['#FFFFFF', '#000000'])
        plot = mpl_renderer.get_plot(g)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), self.weights)
        self.assertEqual(edges.get_clim(), (self.weights.min(), self.weights.max()))

    def test_graph_op_node_color(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, '#000000'), (0, 1, 1, '#FF0000'), (1, 1, 2, '#00FF00')], vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_facecolors(), np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]))

    def test_graph_op_node_color_update(self):
        edges = [(0, 1), (0, 2)]

        def get_graph(i):
            c1, c2, c3 = {0: ('#00FF00', '#0000FF', '#FF0000'), 1: ('#FF0000', '#00FF00', '#0000FF')}[i]
            nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)], vdims='color')
            return Graph((edges, nodes))
        graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_color='color')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_facecolors(), np.array([[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]]))
        plot.update((1,))
        self.assertEqual(artist.get_facecolors(), np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]))

    def test_graph_op_node_color_linear(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 0.5), (0, 1, 1, 1.5), (1, 1, 2, 2.5)], vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0.5, 1.5, 2.5]))
        self.assertEqual(artist.get_clim(), (0.5, 2.5))

    def test_graph_op_node_color_linear_update(self):
        edges = [(0, 1), (0, 2)]

        def get_graph(i):
            c1, c2, c3 = {0: (0.5, 1.5, 2.5), 1: (3, 2, 1)}[i]
            nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)], vdims='color')
            return Graph((edges, nodes))
        graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_color='color', framewise=True)
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0.5, 1.5, 2.5]))
        self.assertEqual(artist.get_clim(), (0.5, 2.5))
        plot.update((1,))
        self.assertEqual(np.asarray(artist.get_array()), np.array([3, 2, 1]))
        self.assertEqual(artist.get_clim(), (1, 3))

    def test_graph_op_node_color_categorical(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 'A'), (0, 1, 1, 'B'), (1, 1, 2, 'A')], vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 0]))

    def test_graph_op_node_size(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 2), (0, 1, 1, 4), (1, 1, 2, 6)], vdims='size')
        graph = Graph((edges, nodes)).opts(node_size='size')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_sizes(), np.array([4, 16, 36]))

    def test_graph_op_node_size_update(self):
        edges = [(0, 1), (0, 2)]

        def get_graph(i):
            c1, c2, c3 = {0: (2, 4, 6), 1: (12, 3, 5)}[i]
            nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)], vdims='size')
            return Graph((edges, nodes))
        graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_size='size')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_sizes(), np.array([4, 16, 36]))
        plot.update((1,))
        self.assertEqual(artist.get_sizes(), np.array([144, 9, 25]))

    def test_graph_op_node_linewidth(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 2), (0, 1, 1, 4), (1, 1, 2, 3.5)], vdims='line_width')
        graph = Graph((edges, nodes)).opts(node_linewidth='line_width')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_linewidths(), [2, 4, 3.5])

    def test_graph_op_node_linewidth_update(self):
        edges = [(0, 1), (0, 2)]

        def get_graph(i):
            c1, c2, c3 = {0: (2, 4, 6), 1: (12, 3, 5)}[i]
            nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)], vdims='line_width')
            return Graph((edges, nodes))
        graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_linewidth='line_width')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_linewidths(), [2, 4, 6])
        plot.update((1,))
        self.assertEqual(artist.get_linewidths(), [12, 3, 5])

    def test_graph_op_node_alpha(self):
        import matplotlib as mpl
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 0.2), (0, 1, 1, 0.6), (1, 1, 2, 1)], vdims='alpha')
        graph = Graph((edges, nodes)).opts(node_alpha='alpha')
        if Version(mpl.__version__) < Version('3.4.0'):
            msg = 'TypeError: alpha must be a float or None'
            with pytest.raises(AbbreviatedException, match=msg):
                mpl_renderer.get_plot(graph)
        else:
            plot = mpl_renderer.get_plot(graph)
            artist = plot.handles['nodes']
            self.assertEqual(artist.get_alpha(), np.array([0.2, 0.6, 1]))

    def test_graph_op_edge_color(self):
        edges = [(0, 1, 'red'), (0, 2, 'green'), (1, 3, 'blue')]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_edgecolors(), np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.50196078, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]))

    def test_graph_op_edge_color_update(self):
        graph = HoloMap({0: Graph([(0, 1, 'red'), (0, 2, 'green'), (1, 3, 'blue')], vdims='color'), 1: Graph([(0, 1, 'green'), (0, 2, 'blue'), (1, 3, 'red')], vdims='color')}).opts(edge_color='color')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_edgecolors(), np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.50196078, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]))
        plot.update((1,))
        self.assertEqual(edges.get_edgecolors(), np.array([[0.0, 0.50196078, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]))

    def test_graph_op_edge_color_linear(self):
        edges = [(0, 1, 2), (0, 2, 0.5), (1, 3, 3)]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), np.array([2, 0.5, 3]))
        self.assertEqual(edges.get_clim(), (0.5, 3))

    def test_graph_op_edge_color_linear_update(self):
        graph = HoloMap({0: Graph([(0, 1, 2), (0, 2, 0.5), (1, 3, 3)], vdims='color'), 1: Graph([(0, 1, 4.3), (0, 2, 1.4), (1, 3, 2.6)], vdims='color')}).opts(edge_color='color', framewise=True)
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), np.array([2, 0.5, 3]))
        self.assertEqual(edges.get_clim(), (0.5, 3))
        plot.update((1,))
        self.assertEqual(np.asarray(edges.get_array()), np.array([4.3, 1.4, 2.6]))
        self.assertEqual(edges.get_clim(), (1.4, 4.3))

    def test_graph_op_edge_color_categorical(self):
        edges = [(0, 1, 'C'), (0, 2, 'B'), (1, 3, 'A')]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), np.array([0, 1, 2]))
        self.assertEqual(edges.get_clim(), (0, 2))

    def test_graph_op_edge_alpha(self):
        edges = [(0, 1, 0.1), (0, 2, 0.5), (1, 3, 0.3)]
        graph = Graph(edges, vdims='alpha').opts(edge_alpha='alpha')
        msg = 'ValueError: Mapping a dimension to the "edge_alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(graph)

    def test_graph_op_edge_linewidth(self):
        edges = [(0, 1, 2), (0, 2, 10), (1, 3, 6)]
        graph = Graph(edges, vdims='line_width').opts(edge_linewidth='line_width')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_linewidths(), [2, 10, 6])

    def test_graph_op_edge_line_width_update(self):
        graph = HoloMap({0: Graph([(0, 1, 2), (0, 2, 0.5), (1, 3, 3)], vdims='line_width'), 1: Graph([(0, 1, 4.3), (0, 2, 1.4), (1, 3, 2.6)], vdims='line_width')}).opts(edge_linewidth='line_width')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_linewidths(), [2, 0.5, 3])
        plot.update((1,))
        self.assertEqual(edges.get_linewidths(), [4.3, 1.4, 2.6])