import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_weight_attribute(self):
    G = nx.Graph()
    G.add_edge(0, 1, weight=1, distance=7)
    G.add_edge(0, 2, weight=30, distance=1)
    G.add_edge(1, 2, weight=1, distance=1)
    G.add_node(3)
    T = nx.minimum_spanning_tree(G, algorithm=self.algo, weight='distance')
    assert nodes_equal(sorted(T), list(range(4)))
    assert edges_equal(sorted(T.edges()), [(0, 2), (1, 2)])
    T = nx.maximum_spanning_tree(G, algorithm=self.algo, weight='distance')
    assert nodes_equal(sorted(T), list(range(4)))
    assert edges_equal(sorted(T.edges()), [(0, 1), (0, 2)])