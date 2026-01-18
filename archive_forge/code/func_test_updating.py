import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_updating(self):
    G2_degree = {n: (in_degree, out_degree) for (n, in_degree), (_, out_degree) in zip(self.G2.in_degree, self.G2.out_degree)}
    gparams, sparams = _initialize_parameters(self.G1, self.G2, G2_degree)
    m, m_rev, T1_out, T1_in, T1_tilde, _, T2_out, T2_in, T2_tilde, _ = sparams
    m[4] = self.mapped[4]
    m_rev[self.mapped[4]] = 4
    _update_Tinout(4, self.mapped[4], gparams, sparams)
    assert T1_out == {5, 9}
    assert T1_in == {3}
    assert T2_out == {'i', 'e'}
    assert T2_in == {'c'}
    assert T1_tilde == {0, 1, 2, 6, 7, 8}
    assert T2_tilde == {'x', 'a', 'b', 'f', 'g', 'h'}
    m[5] = self.mapped[5]
    m_rev[self.mapped[5]] = 5
    _update_Tinout(5, self.mapped[5], gparams, sparams)
    assert T1_out == {9, 8, 7}
    assert T1_in == {3}
    assert T2_out == {'i', 'g', 'h'}
    assert T2_in == {'c'}
    assert T1_tilde == {0, 1, 2, 6}
    assert T2_tilde == {'x', 'a', 'b', 'f'}
    m[6] = self.mapped[6]
    m_rev[self.mapped[6]] = 6
    _update_Tinout(6, self.mapped[6], gparams, sparams)
    assert T1_out == {9, 8, 7}
    assert T1_in == {3, 7}
    assert T2_out == {'i', 'g', 'h'}
    assert T2_in == {'c', 'g'}
    assert T1_tilde == {0, 1, 2}
    assert T2_tilde == {'x', 'a', 'b'}
    m[3] = self.mapped[3]
    m_rev[self.mapped[3]] = 3
    _update_Tinout(3, self.mapped[3], gparams, sparams)
    assert T1_out == {9, 8, 7, 2}
    assert T1_in == {7, 1}
    assert T2_out == {'i', 'g', 'h', 'b'}
    assert T2_in == {'g', 'a'}
    assert T1_tilde == {0}
    assert T2_tilde == {'x'}
    m[0] = self.mapped[0]
    m_rev[self.mapped[0]] = 0
    _update_Tinout(0, self.mapped[0], gparams, sparams)
    assert T1_out == {9, 8, 7, 2}
    assert T1_in == {7, 1}
    assert T2_out == {'i', 'g', 'h', 'b'}
    assert T2_in == {'g', 'a'}
    assert T1_tilde == set()
    assert T2_tilde == set()