import pytest
import networkx as nx
def test_transitivity(self):
    G = nx.Graph()
    assert nx.transitivity(G) == 0