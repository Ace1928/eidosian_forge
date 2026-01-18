import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_number_weakly_connected_components(self):
    for G, C in self.gc:
        U = G.to_undirected()
        w = nx.number_weakly_connected_components(G)
        c = nx.number_connected_components(U)
        assert w == c