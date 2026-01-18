import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_onion_digraph(self):
    G = nx.complete_graph(3)
    G = nx.DiGraph(G)
    G.add_edge(2, 1)
    with pytest.raises(nx.NetworkXNotImplemented, match='not implemented for directed type'):
        nx.onion_layers(G)