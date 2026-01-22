import pytest
import networkx as nx
class TestImmediateDominators:

    def test_exceptions(self):
        G = nx.Graph()
        G.add_node(0)
        pytest.raises(nx.NetworkXNotImplemented, nx.immediate_dominators, G, 0)
        G = nx.MultiGraph(G)
        pytest.raises(nx.NetworkXNotImplemented, nx.immediate_dominators, G, 0)
        G = nx.DiGraph([[0, 0]])
        pytest.raises(nx.NetworkXError, nx.immediate_dominators, G, 1)

    def test_singleton(self):
        G = nx.DiGraph()
        G.add_node(0)
        assert nx.immediate_dominators(G, 0) == {0: 0}
        G.add_edge(0, 0)
        assert nx.immediate_dominators(G, 0) == {0: 0}

    def test_path(self):
        n = 5
        G = nx.path_graph(n, create_using=nx.DiGraph())
        assert nx.immediate_dominators(G, 0) == {i: max(i - 1, 0) for i in range(n)}

    def test_cycle(self):
        n = 5
        G = nx.cycle_graph(n, create_using=nx.DiGraph())
        assert nx.immediate_dominators(G, 0) == {i: max(i - 1, 0) for i in range(n)}

    def test_unreachable(self):
        n = 5
        assert n > 1
        G = nx.path_graph(n, create_using=nx.DiGraph())
        assert nx.immediate_dominators(G, n // 2) == {i: max(i - 1, n // 2) for i in range(n // 2, n)}

    def test_irreducible1(self):
        edges = [(1, 2), (2, 1), (3, 2), (4, 1), (5, 3), (5, 4)]
        G = nx.DiGraph(edges)
        assert nx.immediate_dominators(G, 5) == {i: 5 for i in range(1, 6)}

    def test_irreducible2(self):
        edges = [(1, 2), (2, 1), (2, 3), (3, 2), (4, 2), (4, 3), (5, 1), (6, 4), (6, 5)]
        G = nx.DiGraph(edges)
        result = nx.immediate_dominators(G, 6)
        assert result == {i: 6 for i in range(1, 7)}

    def test_domrel_png(self):
        edges = [(1, 2), (2, 3), (2, 4), (2, 6), (3, 5), (4, 5), (5, 2)]
        G = nx.DiGraph(edges)
        result = nx.immediate_dominators(G, 1)
        assert result == {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2}
        result = nx.immediate_dominators(G.reverse(copy=False), 6)
        assert result == {1: 2, 2: 6, 3: 5, 4: 5, 5: 2, 6: 6}

    def test_boost_example(self):
        edges = [(0, 1), (1, 2), (1, 3), (2, 7), (3, 4), (4, 5), (4, 6), (5, 7), (6, 4)]
        G = nx.DiGraph(edges)
        result = nx.immediate_dominators(G, 0)
        assert result == {0: 0, 1: 0, 2: 1, 3: 1, 4: 3, 5: 4, 6: 4, 7: 1}
        result = nx.immediate_dominators(G.reverse(copy=False), 7)
        assert result == {0: 1, 1: 7, 2: 7, 3: 4, 4: 5, 5: 7, 6: 4, 7: 7}