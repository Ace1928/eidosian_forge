import pytest
import networkx as nx
from networkx.algorithms.bipartite import spectral_bipartivity as sb
def test_star_like(self):
    G = nx.star_graph(2)
    G.add_edge(1, 2)
    assert sb(G) == pytest.approx(0.843, abs=0.001)
    G = nx.star_graph(3)
    G.add_edge(1, 2)
    assert sb(G) == pytest.approx(0.871, abs=0.001)
    G = nx.star_graph(4)
    G.add_edge(1, 2)
    assert sb(G) == pytest.approx(0.89, abs=0.001)