import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
class TestDiGraphISOFeasibility:

    def test_const_covered_neighbors(self):
        G1 = nx.DiGraph([(0, 1), (1, 2), (0, 3), (2, 3)])
        G2 = nx.DiGraph([('a', 'b'), ('b', 'c'), ('a', 'k'), ('c', 'k')])
        gparams = _GraphParameters(G1, G2, None, None, None, None, None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c'}, {'a': 0, 'b': 1, 'c': 2}, None, None, None, None, None, None, None, None)
        u, v = (3, 'k')
        assert _consistent_PT(u, v, gparams, sparams)

    def test_const_no_covered_neighbors(self):
        G1 = nx.DiGraph([(0, 1), (1, 2), (3, 4), (3, 5)])
        G2 = nx.DiGraph([('a', 'b'), ('b', 'c'), ('k', 'w'), ('k', 'z')])
        gparams = _GraphParameters(G1, G2, None, None, None, None, None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c'}, {'a': 0, 'b': 1, 'c': 2}, None, None, None, None, None, None, None, None)
        u, v = (3, 'k')
        assert _consistent_PT(u, v, gparams, sparams)

    def test_const_mixed_covered_uncovered_neighbors(self):
        G1 = nx.DiGraph([(0, 1), (1, 2), (3, 0), (3, 2), (3, 4), (3, 5)])
        G2 = nx.DiGraph([('a', 'b'), ('b', 'c'), ('k', 'a'), ('k', 'c'), ('k', 'w'), ('k', 'z')])
        gparams = _GraphParameters(G1, G2, None, None, None, None, None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c'}, {'a': 0, 'b': 1, 'c': 2}, None, None, None, None, None, None, None, None)
        u, v = (3, 'k')
        assert _consistent_PT(u, v, gparams, sparams)

    def test_const_fail_cases(self):
        G1 = nx.DiGraph([(0, 1), (2, 1), (10, 0), (10, 3), (10, 4), (5, 10), (10, 6), (1, 4), (5, 3)])
        G2 = nx.DiGraph([('a', 'b'), ('c', 'b'), ('k', 'a'), ('k', 'd'), ('k', 'e'), ('f', 'k'), ('k', 'g'), ('b', 'e'), ('f', 'd')])
        gparams = _GraphParameters(G1, G2, None, None, None, None, None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, None, None, None, None, None, None, None, None)
        u, v = (10, 'k')
        assert _consistent_PT(u, v, gparams, sparams)
        G1.remove_node(6)
        assert _consistent_PT(u, v, gparams, sparams)
        G1.add_edge(u, 2)
        assert not _consistent_PT(u, v, gparams, sparams)
        G2.add_edge(v, 'c')
        assert _consistent_PT(u, v, gparams, sparams)
        G2.add_edge(v, 'x')
        G1.add_node(7)
        sparams.mapping.update({7: 'x'})
        sparams.reverse_mapping.update({'x': 7})
        assert not _consistent_PT(u, v, gparams, sparams)
        G1.add_edge(u, 7)
        assert _consistent_PT(u, v, gparams, sparams)

    def test_cut_inconsistent_labels(self):
        G1 = nx.DiGraph([(0, 1), (2, 1), (10, 0), (10, 3), (10, 4), (5, 10), (10, 6), (1, 4), (5, 3)])
        G2 = nx.DiGraph([('a', 'b'), ('c', 'b'), ('k', 'a'), ('k', 'd'), ('k', 'e'), ('f', 'k'), ('k', 'g'), ('b', 'e'), ('f', 'd')])
        l1 = {n: 'blue' for n in G1.nodes()}
        l2 = {n: 'blue' for n in G2.nodes()}
        l1.update({5: 'green'})
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, None, None, None, None, None, None, None, None)
        u, v = (10, 'k')
        assert _cut_PT(u, v, gparams, sparams)

    def test_cut_consistent_labels(self):
        G1 = nx.DiGraph([(0, 1), (2, 1), (10, 0), (10, 3), (10, 4), (5, 10), (10, 6), (1, 4), (5, 3)])
        G2 = nx.DiGraph([('a', 'b'), ('c', 'b'), ('k', 'a'), ('k', 'd'), ('k', 'e'), ('f', 'k'), ('k', 'g'), ('b', 'e'), ('f', 'd')])
        l1 = {n: 'blue' for n in G1.nodes()}
        l2 = {n: 'blue' for n in G2.nodes()}
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, {4}, {5, 10}, {6}, None, {'e'}, {'f', 'k'}, {'g'}, None)
        u, v = (10, 'k')
        assert not _cut_PT(u, v, gparams, sparams)

    def test_cut_same_labels(self):
        G1 = nx.DiGraph([(0, 1), (2, 1), (10, 0), (10, 3), (10, 4), (5, 10), (10, 6), (1, 4), (5, 3)])
        mapped = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 10: 'k'}
        G2 = nx.relabel_nodes(G1, mapped)
        l1 = {n: 'blue' for n in G1.nodes()}
        l2 = {n: 'blue' for n in G2.nodes()}
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, {4}, {5, 10}, {6}, None, {'e'}, {'f', 'k'}, {'g'}, None)
        u, v = (10, 'k')
        assert not _cut_PT(u, v, gparams, sparams)
        G1.remove_edge(u, 4)
        assert _cut_PT(u, v, gparams, sparams)
        G2.remove_edge(v, mapped[4])
        assert not _cut_PT(u, v, gparams, sparams)
        G1.remove_edge(5, u)
        assert _cut_PT(u, v, gparams, sparams)
        G2.remove_edge(mapped[5], v)
        assert not _cut_PT(u, v, gparams, sparams)
        G2.remove_edge(v, mapped[6])
        assert _cut_PT(u, v, gparams, sparams)
        G1.remove_edge(u, 6)
        assert not _cut_PT(u, v, gparams, sparams)
        G1.add_nodes_from([6, 7, 8])
        G2.add_nodes_from(['g', 'y', 'z'])
        sparams.T1_tilde.update({6, 7, 8})
        sparams.T2_tilde.update({'g', 'y', 'z'})
        l1 = {n: 'blue' for n in G1.nodes()}
        l2 = {n: 'blue' for n in G2.nodes()}
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
        assert not _cut_PT(u, v, gparams, sparams)

    def test_cut_different_labels(self):
        G1 = nx.DiGraph([(0, 1), (1, 2), (14, 1), (0, 4), (1, 5), (2, 6), (3, 7), (3, 6), (10, 4), (4, 9), (6, 10), (20, 9), (20, 15), (20, 12), (20, 11), (12, 13), (11, 13), (20, 8), (20, 3), (20, 5), (0, 20)])
        mapped = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 20: 'x'}
        G2 = nx.relabel_nodes(G1, mapped)
        l1 = {n: 'none' for n in G1.nodes()}
        l2 = {}
        l1.update({9: 'blue', 15: 'blue', 12: 'blue', 11: 'green', 3: 'green', 8: 'red', 0: 'red', 5: 'yellow'})
        l2.update({mapped[n]: l for n, l in l1.items()})
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, {4, 5, 6, 7, 20}, {14, 20}, {9, 10, 15, 12, 11, 13, 8}, None, {'e', 'f', 'g', 'x'}, {'o', 'x'}, {'j', 'k', 'l', 'm', 'n', 'i', 'p'}, None)
        u, v = (20, 'x')
        assert not _cut_PT(u, v, gparams, sparams)
        l1.update({9: 'red'})
        assert _cut_PT(u, v, gparams, sparams)
        l2.update({mapped[9]: 'red'})
        assert not _cut_PT(u, v, gparams, sparams)
        G1.add_edge(u, 4)
        assert _cut_PT(u, v, gparams, sparams)
        G2.add_edge(v, mapped[4])
        assert not _cut_PT(u, v, gparams, sparams)
        G1.add_edge(u, 14)
        assert _cut_PT(u, v, gparams, sparams)
        G2.add_edge(v, mapped[14])
        assert not _cut_PT(u, v, gparams, sparams)
        G2.remove_edge(v, mapped[8])
        assert _cut_PT(u, v, gparams, sparams)
        G1.remove_edge(u, 8)
        assert not _cut_PT(u, v, gparams, sparams)
        G1.add_edge(8, 3)
        G2.add_edge(mapped[8], mapped[3])
        sparams.T1.add(8)
        sparams.T2.add(mapped[8])
        sparams.T1_tilde.remove(8)
        sparams.T2_tilde.remove(mapped[8])
        assert not _cut_PT(u, v, gparams, sparams)
        G1.remove_node(5)
        l1.pop(5)
        sparams.T1.remove(5)
        assert _cut_PT(u, v, gparams, sparams)
        G2.remove_node(mapped[5])
        l2.pop(mapped[5])
        sparams.T2.remove(mapped[5])
        assert not _cut_PT(u, v, gparams, sparams)

    def test_predecessor_T1_in_fail(self):
        G1 = nx.DiGraph([(0, 1), (0, 3), (4, 0), (1, 5), (5, 2), (3, 6), (4, 6), (6, 5)])
        mapped = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}
        G2 = nx.relabel_nodes(G1, mapped)
        l1 = {n: 'blue' for n in G1.nodes()}
        l2 = {n: 'blue' for n in G2.nodes()}
        gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
        sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c'}, {'a': 0, 'b': 1, 'c': 2}, {3, 5}, {4, 5}, {6}, None, {'d', 'f'}, {'f'}, {'g'}, None)
        u, v = (6, 'g')
        assert _cut_PT(u, v, gparams, sparams)
        sparams.T2_in.add('e')
        assert not _cut_PT(u, v, gparams, sparams)