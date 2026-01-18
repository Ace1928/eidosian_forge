import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
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