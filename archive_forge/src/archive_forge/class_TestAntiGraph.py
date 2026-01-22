import pytest
import networkx as nx
from networkx.algorithms.approximation import k_components
from networkx.algorithms.approximation.kcomponents import _AntiGraph, _same
class TestAntiGraph:

    @classmethod
    def setup_class(cls):
        cls.Gnp = nx.gnp_random_graph(20, 0.8, seed=42)
        cls.Anp = _AntiGraph(nx.complement(cls.Gnp))
        cls.Gd = nx.davis_southern_women_graph()
        cls.Ad = _AntiGraph(nx.complement(cls.Gd))
        cls.Gk = nx.karate_club_graph()
        cls.Ak = _AntiGraph(nx.complement(cls.Gk))
        cls.GA = [(cls.Gnp, cls.Anp), (cls.Gd, cls.Ad), (cls.Gk, cls.Ak)]

    def test_size(self):
        for G, A in self.GA:
            n = G.order()
            s = len(list(G.edges())) + len(list(A.edges()))
            assert s == n * (n - 1) / 2

    def test_degree(self):
        for G, A in self.GA:
            assert sorted(G.degree()) == sorted(A.degree())

    def test_core_number(self):
        for G, A in self.GA:
            assert nx.core_number(G) == nx.core_number(A)

    def test_connected_components(self):
        for G, A in self.GA:
            gc = [set(c) for c in nx.connected_components(G)]
            ac = [set(c) for c in nx.connected_components(A)]
            for comp in ac:
                assert comp in gc

    def test_adj(self):
        for G, A in self.GA:
            for n, nbrs in G.adj.items():
                a_adj = sorted(((n, sorted(ad)) for n, ad in A.adj.items()))
                g_adj = sorted(((n, sorted(ad)) for n, ad in G.adj.items()))
                assert a_adj == g_adj

    def test_adjacency(self):
        for G, A in self.GA:
            a_adj = list(A.adjacency())
            for n, nbrs in G.adjacency():
                assert (n, set(nbrs)) in a_adj

    def test_neighbors(self):
        for G, A in self.GA:
            node = list(G.nodes())[0]
            assert set(G.neighbors(node)) == set(A.neighbors(node))

    def test_node_not_in_graph(self):
        for G, A in self.GA:
            node = 'non_existent_node'
            pytest.raises(nx.NetworkXError, A.neighbors, node)
            pytest.raises(nx.NetworkXError, G.neighbors, node)

    def test_degree_thingraph(self):
        for G, A in self.GA:
            node = list(G.nodes())[0]
            nodes = list(G.nodes())[1:4]
            assert G.degree(node) == A.degree(node)
            assert sum((d for n, d in G.degree())) == sum((d for n, d in A.degree()))
            assert sum((d for n, d in A.degree())) == sum((d for n, d in A.degree(weight='weight')))
            assert sum((d for n, d in G.degree(nodes))) == sum((d for n, d in A.degree(nodes)))