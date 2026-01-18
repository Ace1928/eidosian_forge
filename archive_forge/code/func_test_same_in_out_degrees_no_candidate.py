import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_same_in_out_degrees_no_candidate(self):
    g1 = nx.DiGraph([(4, 1), (4, 2), (3, 4), (5, 4), (6, 4)])
    g2 = nx.DiGraph([(1, 4), (2, 4), (3, 4), (4, 5), (4, 6)])
    l1 = dict(g1.nodes(data=None, default=-1))
    l2 = dict(g2.nodes(data=None, default=-1))
    gparams = _GraphParameters(g1, g2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups({node: (in_degree, out_degree) for (node, in_degree), (_, out_degree) in zip(g2.in_degree(), g2.out_degree())}))
    g1_degree = {n: (in_degree, out_degree) for (n, in_degree), (_, out_degree) in zip(g1.in_degree, g1.out_degree)}
    m = {1: 1, 2: 2, 3: 3}
    m_rev = m.copy()
    T1_out = {4}
    T1_in = {4}
    T1_tilde = {5, 6}
    T2_out = {4}
    T2_in = {4}
    T2_tilde = {5, 6}
    sparams = _StateParameters(m, m_rev, T1_out, T1_in, T1_tilde, None, T2_out, T2_in, T2_tilde, None)
    u = 4
    candidates = _find_candidates_Di(u, gparams, sparams, g1_degree)
    assert candidates == set()
    assert _find_candidates(u, gparams, sparams, g1_degree) == {4}