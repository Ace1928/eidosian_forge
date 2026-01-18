import pytest
import networkx as nx
def test_random_regular_expander_explicit_construction():
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    G = nx.random_regular_expander_graph(d=4, n=5)
    assert len(G) == 5 and len(G.edges) == 10, 'Should be a complete graph'