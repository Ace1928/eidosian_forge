from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
class FromNetworkXTests(ComparisonTestCase):

    def setUp(self):
        try:
            import networkx as nx
        except ImportError:
            raise SkipTest('Test requires networkx to be installed')

    def test_from_networkx_with_node_attrs(self):
        import networkx as nx
        G = nx.karate_club_graph()
        graph = Graph.from_networkx(G, nx.circular_layout)
        clubs = np.array(['Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Officer', 'Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Mr. Hi', 'Officer', 'Officer', 'Mr. Hi', 'Mr. Hi', 'Officer', 'Mr. Hi', 'Officer', 'Mr. Hi', 'Officer', 'Officer', 'Officer', 'Officer', 'Officer', 'Officer', 'Officer', 'Officer', 'Officer', 'Officer', 'Officer', 'Officer'])
        self.assertEqual(graph.nodes.dimension_values('club'), clubs)

    def test_from_networkx_with_invalid_node_attrs(self):
        import networkx as nx
        FG = nx.Graph()
        FG.add_node(1, test=[])
        FG.add_node(2, test=[])
        FG.add_edge(1, 2)
        graph = Graph.from_networkx(FG, nx.circular_layout)
        self.assertEqual(graph.nodes.vdims, [])
        self.assertEqual(graph.nodes.dimension_values(2), np.array([1, 2]))
        self.assertEqual(graph.array(), np.array([(1, 2)]))

    def test_from_networkx_with_edge_attrs(self):
        import networkx as nx
        FG = nx.Graph()
        FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
        graph = Graph.from_networkx(FG, nx.circular_layout)
        self.assertEqual(graph.dimension_values('weight'), np.array([0.125, 0.75, 1.2, 0.375]))

    def test_from_networkx_with_invalid_edge_attrs(self):
        import networkx as nx
        FG = nx.Graph()
        FG.add_weighted_edges_from([(1, 2, []), (1, 3, []), (2, 4, []), (3, 4, [])])
        graph = Graph.from_networkx(FG, nx.circular_layout)
        self.assertEqual(graph.vdims, [])

    def test_from_networkx_only_nodes(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        graph = Graph.from_networkx(G, nx.circular_layout)
        self.assertEqual(graph.nodes.dimension_values(2), np.array([1, 2, 3]))

    def test_from_networkx_custom_nodes(self):
        import networkx as nx
        FG = nx.Graph()
        FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
        nodes = Dataset([(1, 'A'), (2, 'B'), (3, 'A'), (4, 'B')], 'index', 'some_attribute')
        graph = Graph.from_networkx(FG, nx.circular_layout, nodes=nodes)
        self.assertEqual(graph.nodes.dimension_values('some_attribute'), np.array(['A', 'B', 'A', 'B']))

    def test_from_networkx_dictionary_positions(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        positions = nx.circular_layout(G)
        graph = Graph.from_networkx(G, positions)
        self.assertEqual(graph.nodes.dimension_values(2), np.array([1, 2, 3]))