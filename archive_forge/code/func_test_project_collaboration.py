import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_project_collaboration(self):
    G = nx.Graph()
    G.add_edge('a', 1)
    G.add_edge('b', 1)
    G.add_edge('b', 2)
    G.add_edge('c', 2)
    G.add_edge('c', 3)
    G.add_edge('c', 4)
    G.add_edge('b', 4)
    P = bipartite.collaboration_weighted_projected_graph(G, 'abc')
    assert P['a']['b']['weight'] == 1
    assert P['b']['c']['weight'] == 2