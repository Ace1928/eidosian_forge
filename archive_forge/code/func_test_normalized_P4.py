import pytest
import networkx as nx
def test_normalized_P4(self):
    """Edge betweenness centrality: P4"""
    G = nx.path_graph(4)
    b = nx.edge_betweenness_centrality(G, weight=None, normalized=True)
    b_answer = {(0, 1): 3, (1, 2): 4, (2, 3): 3}
    for n in sorted(G.edges()):
        assert b[n] == pytest.approx(b_answer[n] / 6, abs=1e-07)