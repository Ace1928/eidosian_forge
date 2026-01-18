import pytest
import networkx as nx
def test_weighted_graph(self):
    """Edge betweenness centrality: weighted"""
    eList = [(0, 1, 5), (0, 2, 4), (0, 3, 3), (0, 4, 2), (1, 2, 4), (1, 3, 1), (1, 4, 3), (2, 4, 5), (3, 4, 4)]
    G = nx.Graph()
    G.add_weighted_edges_from(eList)
    b = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
    b_answer = {(0, 1): 0.0, (0, 2): 1.0, (0, 3): 2.0, (0, 4): 1.0, (1, 2): 2.0, (1, 3): 3.5, (1, 4): 1.5, (2, 4): 1.0, (3, 4): 0.5}
    for n in sorted(G.edges()):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)