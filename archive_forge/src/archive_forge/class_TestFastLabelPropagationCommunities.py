from itertools import chain, combinations
import pytest
import networkx as nx
class TestFastLabelPropagationCommunities:
    N = 100
    K = 15

    def _check_communities(self, G, truth, weight=None, seed=None):
        C = nx.community.fast_label_propagation_communities(G, weight=weight, seed=seed)
        assert {frozenset(c) for c in C} == truth

    def test_null_graph(self):
        G = nx.null_graph()
        truth = set()
        self._check_communities(G, truth)

    def test_empty_graph(self):
        G = nx.empty_graph(self.N)
        truth = {frozenset([i]) for i in G}
        self._check_communities(G, truth)

    def test_star_graph(self):
        G = nx.star_graph(self.N)
        truth = {frozenset(G)}
        self._check_communities(G, truth)

    def test_complete_graph(self):
        G = nx.complete_graph(self.N)
        truth = {frozenset(G)}
        self._check_communities(G, truth)

    def test_bipartite_graph(self):
        G = nx.complete_bipartite_graph(self.N // 2, self.N // 2)
        truth = {frozenset(G)}
        self._check_communities(G, truth)

    def test_random_graph(self):
        G = nx.gnm_random_graph(self.N, self.N * self.K // 2)
        truth = {frozenset(G)}
        self._check_communities(G, truth)

    def test_disjoin_cliques(self):
        G = nx.Graph(['ab', 'AB', 'AC', 'BC', '12', '13', '14', '23', '24', '34'])
        truth = {frozenset('ab'), frozenset('ABC'), frozenset('1234')}
        self._check_communities(G, truth)

    def test_ring_of_cliques(self):
        G = nx.ring_of_cliques(self.N, self.K)
        truth = {frozenset([self.K * i + k for k in range(self.K)]) for i in range(self.N)}
        self._check_communities(G, truth)

    def test_larger_graph(self):
        G = nx.gnm_random_graph(100 * self.N, 50 * self.N * self.K)
        nx.community.fast_label_propagation_communities(G)

    def test_graph_type(self):
        G1 = nx.complete_graph(self.N, nx.MultiDiGraph())
        G2 = nx.MultiGraph(G1)
        G3 = nx.DiGraph(G1)
        G4 = nx.Graph(G1)
        truth = {frozenset(G1)}
        self._check_communities(G1, truth)
        self._check_communities(G2, truth)
        self._check_communities(G3, truth)
        self._check_communities(G4, truth)

    def test_weight_argument(self):
        G = nx.MultiDiGraph()
        G.add_edge(1, 2, weight=1.41)
        G.add_edge(2, 1, weight=1.41)
        G.add_edge(2, 3)
        G.add_edge(3, 4, weight=3.14)
        truth = {frozenset({1, 2}), frozenset({3, 4})}
        self._check_communities(G, truth, weight='weight')

    def test_seed_argument(self):
        G = nx.karate_club_graph()
        C = nx.community.fast_label_propagation_communities(G, seed=2023)
        truth = {frozenset(c) for c in C}
        self._check_communities(G, truth, seed=2023)