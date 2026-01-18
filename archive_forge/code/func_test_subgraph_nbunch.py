import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_subgraph_nbunch(self):
    nullgraph = nx.null_graph()
    K1 = nx.complete_graph(1)
    K3 = nx.complete_graph(3)
    K5 = nx.complete_graph(5)
    H = K5.subgraph(1)
    assert nx.is_isomorphic(H, K1)
    H = K5.subgraph({1})
    assert nx.is_isomorphic(H, K1)
    H = K5.subgraph(iter(K3))
    assert nx.is_isomorphic(H, K3)
    H = K5.subgraph(K3)
    assert nx.is_isomorphic(H, K3)
    H = K5.subgraph([9])
    assert nx.is_isomorphic(H, nullgraph)