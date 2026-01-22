import pytest
import networkx as nx
class TestTriangles:

    def test_empty(self):
        G = nx.Graph()
        assert list(nx.triangles(G).values()) == []

    def test_path(self):
        G = nx.path_graph(10)
        assert list(nx.triangles(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.triangles(G) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def test_cubical(self):
        G = nx.cubical_graph()
        assert list(nx.triangles(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.triangles(G, 1) == 0
        assert list(nx.triangles(G, [1, 2]).values()) == [0, 0]
        assert nx.triangles(G, 1) == 0
        assert nx.triangles(G, [1, 2]) == {1: 0, 2: 0}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert list(nx.triangles(G).values()) == [6, 6, 6, 6, 6]
        assert sum(nx.triangles(G).values()) / 3 == 10
        assert nx.triangles(G, 1) == 6
        G.remove_edge(1, 2)
        assert list(nx.triangles(G).values()) == [5, 3, 3, 5, 5]
        assert nx.triangles(G, 1) == 3
        G.add_edge(3, 3)
        assert list(nx.triangles(G).values()) == [5, 3, 3, 5, 5]
        assert nx.triangles(G, 3) == 5