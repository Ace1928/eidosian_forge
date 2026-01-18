import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_restoring(self):
    m = {0: 'x', 3: 'c', 4: 'd', 5: 'e', 6: 'f'}
    m_rev = {'x': 0, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    T1_out = {2, 7, 9, 8}
    T1_in = {1, 7}
    T2_out = {'b', 'g', 'i', 'h'}
    T2_in = {'a', 'g'}
    T1_tilde = set()
    T2_tilde = set()
    gparams = _GraphParameters(self.G1, self.G2, {}, {}, {}, {}, {})
    sparams = _StateParameters(m, m_rev, T1_out, T1_in, T1_tilde, None, T2_out, T2_in, T2_tilde, None)
    m.pop(0)
    m_rev.pop('x')
    _restore_Tinout_Di(0, self.mapped[0], gparams, sparams)
    assert T1_out == {2, 7, 9, 8}
    assert T1_in == {1, 7}
    assert T2_out == {'b', 'g', 'i', 'h'}
    assert T2_in == {'a', 'g'}
    assert T1_tilde == {0}
    assert T2_tilde == {'x'}
    m.pop(6)
    m_rev.pop('f')
    _restore_Tinout_Di(6, self.mapped[6], gparams, sparams)
    assert T1_out == {2, 9, 8, 7}
    assert T1_in == {1}
    assert T2_out == {'b', 'i', 'h', 'g'}
    assert T2_in == {'a'}
    assert T1_tilde == {0, 6}
    assert T2_tilde == {'x', 'f'}
    m.pop(3)
    m_rev.pop('c')
    _restore_Tinout_Di(3, self.mapped[3], gparams, sparams)
    assert T1_out == {9, 8, 7}
    assert T1_in == {3}
    assert T2_out == {'i', 'h', 'g'}
    assert T2_in == {'c'}
    assert T1_tilde == {0, 6, 1, 2}
    assert T2_tilde == {'x', 'f', 'a', 'b'}
    m.pop(5)
    m_rev.pop('e')
    _restore_Tinout_Di(5, self.mapped[5], gparams, sparams)
    assert T1_out == {9, 5}
    assert T1_in == {3}
    assert T2_out == {'i', 'e'}
    assert T2_in == {'c'}
    assert T1_tilde == {0, 6, 1, 2, 8, 7}
    assert T2_tilde == {'x', 'f', 'a', 'b', 'h', 'g'}
    m.pop(4)
    m_rev.pop('d')
    _restore_Tinout_Di(4, self.mapped[4], gparams, sparams)
    assert T1_out == set()
    assert T1_in == set()
    assert T2_out == set()
    assert T2_in == set()
    assert T1_tilde == set(self.G1.nodes())
    assert T2_tilde == set(self.G2.nodes())