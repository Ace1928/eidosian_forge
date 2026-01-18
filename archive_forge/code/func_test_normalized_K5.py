import pytest
import networkx as nx
def test_normalized_K5(self):
    """Edge betweenness centrality: K5"""
    G = nx.complete_graph(5)
    b = nx.edge_betweenness_centrality(G, weight=None, normalized=True)
    b_answer = dict.fromkeys(G.edges(), 1 / 10)
    for n in sorted(G.edges()):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)