import pytest
import networkx as nx
def test_normalized_p1(self):
    """
        Edge betweenness subset centrality: P1
        if n <= 1: no normalization b=0 for all nodes
        """
    G = nx.Graph()
    nx.add_path(G, range(1))
    b_answer = dict.fromkeys(G.edges(), 0)
    b = nx.edge_betweenness_centrality_subset(G, sources=[0], targets=[0], normalized=True, weight=None)
    for n in G.edges():
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)