import pytest
import networkx as nx
class TestSubsetBetweennessCentrality:

    def test_K5(self):
        """Betweenness Centrality Subset: K5"""
        G = nx.complete_graph(5)
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[1, 3], weight=None)
        b_answer = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P5_directed(self):
        """Betweenness Centrality Subset: P5 directed"""
        G = nx.DiGraph()
        nx.add_path(G, range(5))
        b_answer = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P5(self):
        """Betweenness Centrality Subset: P5"""
        G = nx.Graph()
        nx.add_path(G, range(5))
        b_answer = {0: 0, 1: 0.5, 2: 0.5, 3: 0, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P5_multiple_target(self):
        """Betweenness Centrality Subset: P5 multiple target"""
        G = nx.Graph()
        nx.add_path(G, range(5))
        b_answer = {0: 0, 1: 1, 2: 1, 3: 0.5, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3, 4], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_box(self):
        """Betweenness Centrality Subset: box"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        b_answer = {0: 0, 1: 0.25, 2: 0.25, 3: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_box_and_path(self):
        """Betweenness Centrality Subset: box and path"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
        b_answer = {0: 0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3, 4], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_box_and_path2(self):
        """Betweenness Centrality Subset: box and path multiple target"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 20), (20, 3), (3, 4)])
        b_answer = {0: 0, 1: 1.0, 2: 0.5, 20: 0.5, 3: 0.5, 4: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3, 4], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_diamond_multi_path(self):
        """Betweenness Centrality Subset: Diamond Multi Path"""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (1, 10), (10, 11), (11, 12), (12, 9), (2, 6), (3, 6), (4, 6), (5, 7), (7, 8), (6, 8), (8, 9)])
        b = nx.betweenness_centrality_subset(G, sources=[1], targets=[9], weight=None)
        expected_b = {1: 0, 2: 1.0 / 10, 3: 1.0 / 10, 4: 1.0 / 10, 5: 1.0 / 10, 6: 3.0 / 10, 7: 1.0 / 10, 8: 4.0 / 10, 9: 0, 10: 1.0 / 10, 11: 1.0 / 10, 12: 1.0 / 10}
        for n in sorted(G):
            assert b[n] == pytest.approx(expected_b[n], abs=1e-07)

    def test_normalized_p2(self):
        """
        Betweenness Centrality Subset: Normalized P2
        if n <= 2:  no normalization, betweenness centrality should be 0 for all nodes.
        """
        G = nx.Graph()
        nx.add_path(G, range(2))
        b_answer = {0: 0, 1: 0.0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[1], normalized=True, weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_normalized_P5_directed(self):
        """Betweenness Centrality Subset: Normalized Directed P5"""
        G = nx.DiGraph()
        nx.add_path(G, range(5))
        b_answer = {0: 0, 1: 1.0 / 12.0, 2: 1.0 / 12.0, 3: 0, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3], normalized=True, weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_weighted_graph(self):
        """Betweenness Centrality Subset: Weighted Graph"""
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=3)
        G.add_edge(0, 2, weight=2)
        G.add_edge(0, 3, weight=6)
        G.add_edge(0, 4, weight=4)
        G.add_edge(1, 3, weight=5)
        G.add_edge(1, 5, weight=5)
        G.add_edge(2, 4, weight=1)
        G.add_edge(3, 4, weight=2)
        G.add_edge(3, 5, weight=1)
        G.add_edge(4, 5, weight=4)
        b_answer = {0: 0.0, 1: 0.0, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[5], normalized=False, weight='weight')
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)