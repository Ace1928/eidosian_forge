import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
class TestGraphCandidateSelection:
    G1_edges = [(1, 2), (1, 4), (1, 5), (2, 3), (2, 4), (3, 4), (4, 5), (1, 6), (6, 7), (6, 8), (8, 9), (7, 9)]
    mapped = {0: 'x', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i'}

    def test_no_covered_neighbors_no_labels(self):
        G1 = nx.Graph()
        G1.add_edges_from(self.G1_edges)
        G1.add_node(0)
        G2 = nx.relabel_nodes(G1, self.mapped)
        G1_degree = dict(G1.degree)
        l1 = dict(G1.nodes(data='label', default=-1))
        l2 = dict(G2.nodes(data='label', default=-1))
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
        m = {9: self.mapped[9], 1: self.mapped[1]}
        m_rev = {self.mapped[9]: 9, self.mapped[1]: 1}
        T1 = {7, 8, 2, 4, 5}
        T1_tilde = {0, 3, 6}
        T2 = {'g', 'h', 'b', 'd', 'e'}
        T2_tilde = {'x', 'c', 'f'}
        sparams = _StateParameters(m, m_rev, T1, None, T1_tilde, None, T2, None, T2_tilde, None)
        u = 3
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u]}
        u = 0
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u]}
        m.pop(9)
        m_rev.pop(self.mapped[9])
        T1 = {2, 4, 5, 6}
        T1_tilde = {0, 3, 7, 8, 9}
        T2 = {'g', 'h', 'b', 'd', 'e', 'f'}
        T2_tilde = {'x', 'c', 'g', 'h', 'i'}
        sparams = _StateParameters(m, m_rev, T1, None, T1_tilde, None, T2, None, T2_tilde, None)
        u = 7
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u], self.mapped[8], self.mapped[3], self.mapped[9]}

    def test_no_covered_neighbors_with_labels(self):
        G1 = nx.Graph()
        G1.add_edges_from(self.G1_edges)
        G1.add_node(0)
        G2 = nx.relabel_nodes(G1, self.mapped)
        G1_degree = dict(G1.degree)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
        nx.set_node_attributes(G2, dict(zip([self.mapped[n] for n in G1], it.cycle(labels_many))), 'label')
        l1 = dict(G1.nodes(data='label', default=-1))
        l2 = dict(G2.nodes(data='label', default=-1))
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
        m = {9: self.mapped[9], 1: self.mapped[1]}
        m_rev = {self.mapped[9]: 9, self.mapped[1]: 1}
        T1 = {7, 8, 2, 4, 5, 6}
        T1_tilde = {0, 3}
        T2 = {'g', 'h', 'b', 'd', 'e', 'f'}
        T2_tilde = {'x', 'c'}
        sparams = _StateParameters(m, m_rev, T1, None, T1_tilde, None, T2, None, T2_tilde, None)
        u = 3
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u]}
        u = 0
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u]}
        G1.nodes[u]['label'] = 'blue'
        l1 = dict(G1.nodes(data='label', default=-1))
        l2 = dict(G2.nodes(data='label', default=-1))
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == set()
        m.pop(9)
        m_rev.pop(self.mapped[9])
        T1 = {2, 4, 5, 6}
        T1_tilde = {0, 3, 7, 8, 9}
        T2 = {'b', 'd', 'e', 'f'}
        T2_tilde = {'x', 'c', 'g', 'h', 'i'}
        sparams = _StateParameters(m, m_rev, T1, None, T1_tilde, None, T2, None, T2_tilde, None)
        u = 7
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u]}
        G1.nodes[8]['label'] = G1.nodes[7]['label']
        G2.nodes[self.mapped[8]]['label'] = G1.nodes[7]['label']
        l1 = dict(G1.nodes(data='label', default=-1))
        l2 = dict(G2.nodes(data='label', default=-1))
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u], self.mapped[8]}

    def test_covered_neighbors_no_labels(self):
        G1 = nx.Graph()
        G1.add_edges_from(self.G1_edges)
        G1.add_node(0)
        G2 = nx.relabel_nodes(G1, self.mapped)
        G1_degree = dict(G1.degree)
        l1 = dict(G1.nodes(data=None, default=-1))
        l2 = dict(G2.nodes(data=None, default=-1))
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
        m = {9: self.mapped[9], 1: self.mapped[1]}
        m_rev = {self.mapped[9]: 9, self.mapped[1]: 1}
        T1 = {7, 8, 2, 4, 5, 6}
        T1_tilde = {0, 3}
        T2 = {'g', 'h', 'b', 'd', 'e', 'f'}
        T2_tilde = {'x', 'c'}
        sparams = _StateParameters(m, m_rev, T1, None, T1_tilde, None, T2, None, T2_tilde, None)
        u = 5
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u]}
        u = 6
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u], self.mapped[2]}

    def test_covered_neighbors_with_labels(self):
        G1 = nx.Graph()
        G1.add_edges_from(self.G1_edges)
        G1.add_node(0)
        G2 = nx.relabel_nodes(G1, self.mapped)
        G1_degree = dict(G1.degree)
        nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
        nx.set_node_attributes(G2, dict(zip([self.mapped[n] for n in G1], it.cycle(labels_many))), 'label')
        l1 = dict(G1.nodes(data='label', default=-1))
        l2 = dict(G2.nodes(data='label', default=-1))
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
        m = {9: self.mapped[9], 1: self.mapped[1]}
        m_rev = {self.mapped[9]: 9, self.mapped[1]: 1}
        T1 = {7, 8, 2, 4, 5, 6}
        T1_tilde = {0, 3}
        T2 = {'g', 'h', 'b', 'd', 'e', 'f'}
        T2_tilde = {'x', 'c'}
        sparams = _StateParameters(m, m_rev, T1, None, T1_tilde, None, T2, None, T2_tilde, None)
        u = 5
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u]}
        u = 6
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u]}
        G1.nodes[2]['label'] = G1.nodes[u]['label']
        G2.nodes[self.mapped[2]]['label'] = G1.nodes[u]['label']
        l1 = dict(G1.nodes(data='label', default=-1))
        l2 = dict(G2.nodes(data='label', default=-1))
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
        candidates = _find_candidates(u, gparams, sparams, G1_degree)
        assert candidates == {self.mapped[u], self.mapped[2]}