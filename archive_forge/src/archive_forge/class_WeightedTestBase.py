import pytest
import networkx as nx
from networkx.utils import pairwise
class WeightedTestBase:
    """Base class for test classes that test functions for computing
    shortest paths in weighted graphs.

    """

    def setup_method(self):
        """Creates some graphs for use in the unit tests."""
        cnlti = nx.convert_node_labels_to_integers
        self.grid = cnlti(nx.grid_2d_graph(4, 4), first_label=1, ordering='sorted')
        self.cycle = nx.cycle_graph(7)
        self.directed_cycle = nx.cycle_graph(7, create_using=nx.DiGraph())
        self.XG = nx.DiGraph()
        self.XG.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('y', 's', 7), ('y', 'v', 6)])
        self.MXG = nx.MultiDiGraph(self.XG)
        self.MXG.add_edge('s', 'u', weight=15)
        self.XG2 = nx.DiGraph()
        self.XG2.add_weighted_edges_from([[1, 4, 1], [4, 5, 1], [5, 6, 1], [6, 3, 1], [1, 3, 50], [1, 2, 100], [2, 3, 100]])
        self.XG3 = nx.Graph()
        self.XG3.add_weighted_edges_from([[0, 1, 2], [1, 2, 12], [2, 3, 1], [3, 4, 5], [4, 5, 1], [5, 0, 10]])
        self.XG4 = nx.Graph()
        self.XG4.add_weighted_edges_from([[0, 1, 2], [1, 2, 2], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1], [6, 7, 1], [7, 0, 1]])
        self.MXG4 = nx.MultiGraph(self.XG4)
        self.MXG4.add_edge(0, 1, weight=3)
        self.G = nx.DiGraph()
        self.G.add_edges_from([('s', 'u'), ('s', 'x'), ('u', 'v'), ('u', 'x'), ('v', 'y'), ('x', 'u'), ('x', 'v'), ('x', 'y'), ('y', 's'), ('y', 'v')])