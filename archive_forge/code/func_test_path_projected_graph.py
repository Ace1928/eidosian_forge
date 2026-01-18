import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_path_projected_graph(self):
    G = nx.path_graph(4)
    P = bipartite.projected_graph(G, [1, 3])
    assert nodes_equal(list(P), [1, 3])
    assert edges_equal(list(P.edges()), [(1, 3)])
    P = bipartite.projected_graph(G, [0, 2])
    assert nodes_equal(list(P), [0, 2])
    assert edges_equal(list(P.edges()), [(0, 2)])
    G = nx.MultiGraph([(0, 1)])
    with pytest.raises(nx.NetworkXError, match='not defined for multigraphs'):
        bipartite.projected_graph(G, [0])