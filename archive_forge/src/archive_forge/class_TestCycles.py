from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
class TestCycles:

    @classmethod
    def setup_class(cls):
        G = nx.Graph()
        nx.add_cycle(G, [0, 1, 2, 3])
        nx.add_cycle(G, [0, 3, 4, 5])
        nx.add_cycle(G, [0, 1, 6, 7, 8])
        G.add_edge(8, 9)
        cls.G = G

    def is_cyclic_permutation(self, a, b):
        n = len(a)
        if len(b) != n:
            return False
        l = a + a
        return any((l[i:i + n] == b for i in range(n)))

    def test_cycle_basis(self):
        G = self.G
        cy = nx.cycle_basis(G, 0)
        sort_cy = sorted((sorted(c) for c in cy))
        assert sort_cy == [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]]
        cy = nx.cycle_basis(G, 1)
        sort_cy = sorted((sorted(c) for c in cy))
        assert sort_cy == [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]]
        cy = nx.cycle_basis(G, 9)
        sort_cy = sorted((sorted(c) for c in cy))
        assert sort_cy == [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]]
        nx.add_cycle(G, 'ABC')
        cy = nx.cycle_basis(G, 9)
        sort_cy = sorted((sorted(c) for c in cy[:-1])) + [sorted(cy[-1])]
        assert sort_cy == [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5], ['A', 'B', 'C']]

    def test_cycle_basis2(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            G = nx.DiGraph()
            cy = nx.cycle_basis(G, 0)

    def test_cycle_basis3(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            G = nx.MultiGraph()
            cy = nx.cycle_basis(G, 0)

    def test_cycle_basis_ordered(self):
        G = nx.cycle_graph(5)
        G.update(nx.cycle_graph(range(3, 8)))
        cbG = nx.cycle_basis(G)
        perm = {1: 0, 0: 1}
        H = nx.relabel_nodes(G, perm)
        cbH = [[perm.get(n, n) for n in cyc] for cyc in nx.cycle_basis(H)]
        assert cbG == cbH

    def test_cycle_basis_self_loop(self):
        """Tests the function for graphs with self loops"""
        G = nx.Graph()
        nx.add_cycle(G, [0, 1, 2, 3])
        nx.add_cycle(G, [0, 0, 6, 2])
        cy = nx.cycle_basis(G)
        sort_cy = sorted((sorted(c) for c in cy))
        assert sort_cy == [[0], [0, 1, 2], [0, 2, 3], [0, 2, 6]]

    def test_simple_cycles(self):
        edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
        G = nx.DiGraph(edges)
        cc = sorted(nx.simple_cycles(G))
        ca = [[0], [0, 1, 2], [0, 2], [1, 2], [2]]
        assert len(cc) == len(ca)
        for c in cc:
            assert any((self.is_cyclic_permutation(c, rc) for rc in ca))

    def test_unsortable(self):
        G = nx.DiGraph()
        nx.add_cycle(G, ['a', 1])
        c = list(nx.simple_cycles(G))
        assert len(c) == 1

    def test_simple_cycles_small(self):
        G = nx.DiGraph()
        nx.add_cycle(G, [1, 2, 3])
        c = sorted(nx.simple_cycles(G))
        assert len(c) == 1
        assert self.is_cyclic_permutation(c[0], [1, 2, 3])
        nx.add_cycle(G, [10, 20, 30])
        cc = sorted(nx.simple_cycles(G))
        assert len(cc) == 2
        ca = [[1, 2, 3], [10, 20, 30]]
        for c in cc:
            assert any((self.is_cyclic_permutation(c, rc) for rc in ca))

    def test_simple_cycles_empty(self):
        G = nx.DiGraph()
        assert list(nx.simple_cycles(G)) == []

    def worst_case_graph(self, k):
        G = nx.DiGraph()
        for n in range(2, k + 2):
            G.add_edge(1, n)
            G.add_edge(n, k + 2)
        G.add_edge(2 * k + 1, 1)
        for n in range(k + 2, 2 * k + 2):
            G.add_edge(n, 2 * k + 2)
            G.add_edge(n, n + 1)
        G.add_edge(2 * k + 3, k + 2)
        for n in range(2 * k + 3, 3 * k + 3):
            G.add_edge(2 * k + 2, n)
            G.add_edge(n, 3 * k + 3)
        G.add_edge(3 * k + 3, 2 * k + 2)
        return G

    def test_worst_case_graph(self):
        for k in range(3, 10):
            G = self.worst_case_graph(k)
            l = len(list(nx.simple_cycles(G)))
            assert l == 3 * k

    def test_recursive_simple_and_not(self):
        for k in range(2, 10):
            G = self.worst_case_graph(k)
            cc = sorted(nx.simple_cycles(G))
            rcc = sorted(nx.recursive_simple_cycles(G))
            assert len(cc) == len(rcc)
            for c in cc:
                assert any((self.is_cyclic_permutation(c, r) for r in rcc))
            for rc in rcc:
                assert any((self.is_cyclic_permutation(rc, c) for c in cc))

    def test_simple_graph_with_reported_bug(self):
        G = nx.DiGraph()
        edges = [(0, 2), (0, 3), (1, 0), (1, 3), (2, 1), (2, 4), (3, 2), (3, 4), (4, 0), (4, 1), (4, 5), (5, 0), (5, 1), (5, 2), (5, 3)]
        G.add_edges_from(edges)
        cc = sorted(nx.simple_cycles(G))
        assert len(cc) == 26
        rcc = sorted(nx.recursive_simple_cycles(G))
        assert len(cc) == len(rcc)
        for c in cc:
            assert any((self.is_cyclic_permutation(c, rc) for rc in rcc))
        for rc in rcc:
            assert any((self.is_cyclic_permutation(rc, c) for c in cc))