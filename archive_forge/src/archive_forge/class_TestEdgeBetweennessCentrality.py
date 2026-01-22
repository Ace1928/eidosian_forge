import pytest
import networkx as nx
class TestEdgeBetweennessCentrality:

    def test_K5(self):
        """Edge betweenness centrality: K5"""
        G = nx.complete_graph(5)
        b = nx.edge_betweenness_centrality(G, weight=None, normalized=False)
        b_answer = dict.fromkeys(G.edges(), 1)
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_normalized_K5(self):
        """Edge betweenness centrality: K5"""
        G = nx.complete_graph(5)
        b = nx.edge_betweenness_centrality(G, weight=None, normalized=True)
        b_answer = dict.fromkeys(G.edges(), 1 / 10)
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_C4(self):
        """Edge betweenness centrality: C4"""
        G = nx.cycle_graph(4)
        b = nx.edge_betweenness_centrality(G, weight=None, normalized=True)
        b_answer = {(0, 1): 2, (0, 3): 2, (1, 2): 2, (2, 3): 2}
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n] / 6, abs=1e-07)

    def test_P4(self):
        """Edge betweenness centrality: P4"""
        G = nx.path_graph(4)
        b = nx.edge_betweenness_centrality(G, weight=None, normalized=False)
        b_answer = {(0, 1): 3, (1, 2): 4, (2, 3): 3}
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_normalized_P4(self):
        """Edge betweenness centrality: P4"""
        G = nx.path_graph(4)
        b = nx.edge_betweenness_centrality(G, weight=None, normalized=True)
        b_answer = {(0, 1): 3, (1, 2): 4, (2, 3): 3}
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n] / 6, abs=1e-07)

    def test_balanced_tree(self):
        """Edge betweenness centrality: balanced tree"""
        G = nx.balanced_tree(r=2, h=2)
        b = nx.edge_betweenness_centrality(G, weight=None, normalized=False)
        b_answer = {(0, 1): 12, (0, 2): 12, (1, 3): 6, (1, 4): 6, (2, 5): 6, (2, 6): 6}
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)