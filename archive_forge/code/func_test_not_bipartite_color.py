import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_not_bipartite_color(self):
    with pytest.raises(nx.NetworkXError):
        c = bipartite.color(nx.complete_graph(4))