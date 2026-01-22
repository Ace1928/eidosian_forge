import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def _model3_nd_sets_normalizeflatten(self):
    m = ConcreteModel()
    m.d1 = Set(initialize=[1, 2])
    m.d2 = Set(initialize=[('a', 1), ('b', 2)])
    m.dn = Set(initialize=[('c', 3), ('d', 4, 5)], dimen=None)
    m.v_2n = Var(m.d2, m.dn)
    m.v_12 = Var(m.d1, m.d2)
    m.v_212 = Var(m.d2, m.d1, m.d2)
    m.v_12n = Var(m.d1, m.d2, m.dn)
    m.v_1n2n = Var(m.d1, m.dn, m.d2, m.dn)
    m.b = Block(m.d1, m.d2, m.dn)
    for i1 in m.d1:
        for i2 in m.d2:
            for i_n in m.dn:
                m.b[i1, i2, i_n].v0 = Var()
                m.b[i1, i2, i_n].v1 = Var(m.d1)
                m.b[i1, i2, i_n].v2 = Var(m.d2)
                m.b[i1, i2, i_n].vn = Var(m.dn)
    return m