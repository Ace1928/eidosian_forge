import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_null_subgraph(self):
    nullgraph = nx.null_graph()
    G = nx.null_graph()
    H = G.subgraph([])
    assert nx.is_isomorphic(H, nullgraph)