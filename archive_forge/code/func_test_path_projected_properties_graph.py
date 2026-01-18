import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_path_projected_properties_graph(self):
    G = nx.path_graph(4)
    G.add_node(1, name='one')
    G.add_node(2, name='two')
    P = bipartite.projected_graph(G, [1, 3])
    assert nodes_equal(list(P), [1, 3])
    assert edges_equal(list(P.edges()), [(1, 3)])
    assert P.nodes[1]['name'] == G.nodes[1]['name']
    P = bipartite.projected_graph(G, [0, 2])
    assert nodes_equal(list(P), [0, 2])
    assert edges_equal(list(P.edges()), [(0, 2)])
    assert P.nodes[2]['name'] == G.nodes[2]['name']