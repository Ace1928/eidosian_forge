import pytest
import networkx as nx
def test_sample_from_P3(self):
    """Betweenness centrality: P3 sample"""
    G = nx.path_graph(3)
    b_answer = {0: 0.0, 1: 1.0, 2: 0.0}
    b = nx.betweenness_centrality(G, k=3, weight=None, normalized=False, seed=1)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
    b = nx.betweenness_centrality(G, k=2, weight=None, normalized=False, seed=1)
    b_approx1 = {0: 0.0, 1: 1.5, 2: 0.0}
    b_approx2 = {0: 0.0, 1: 0.75, 2: 0.0}
    for n in sorted(G):
        assert b[n] in (b_approx1[n], b_approx2[n])