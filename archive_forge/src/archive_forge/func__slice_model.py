import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def _slice_model(self):
    m = ConcreteModel()
    m.d1_1 = Set(initialize=[1, 2, 3])
    m.d1_2 = Set(initialize=['a', 'b', 'c'])
    m.d1_3 = Set(initialize=[1.1, 1.2, 1.3])
    m.d2 = Set(initialize=[('a', 1), ('b', 2)])
    m.dn = Set(initialize=[('c', 3), ('d', 4, 5)], dimen=None)

    @m.Block()
    def b(b):
        b.b = Block()

        @b.Block(m.d1_1)
        def b1(b1, i):
            b1.v = Var()
            b1.v1 = Var(m.d1_3)
            b1.v2 = Var(m.d1_1, m.d1_2)
            b1.vn = Var(m.dn, m.d1_2)

        @b.Block(m.d1_1, m.d1_2)
        def b2(b2, i, j):
            b2.v = Var()
            b2.v1 = Var(m.d1_3)
            b2.v2 = Var(m.d1_1, m.d1_2)
            b2.vn = Var(m.d1_1, m.dn, m.d1_2)

        @b.Block(m.d1_3, m.d2)
        def b3(b3, i, j, k):
            b3.v = Var()
            b3.v1 = Var(m.d1_3)
            b3.v2 = Var(m.d1_1, m.d1_2)
            b3.vn = Var(m.d1_1, m.dn, m.d2)
        b.bn = Block(m.d1_2, m.dn, m.d2)
        b.bn['a', 'c', 3, 'a', 1].v = Var()
        b.bn['a', 'c', 3, 'a', 1].v1 = Var(m.d1_3)
        b.bn['a', 'c', 3, 'a', 1].v2 = Var(m.d1_1, m.d1_2)
        b.bn['a', 'c', 3, 'a', 1].vn = Var(m.d1_1, m.dn, m.d2)
        b.bn['a', 'd', 4, 5, 'a', 1].v = Var()
        b.bn['a', 'd', 4, 5, 'a', 1].v1 = Var(m.d1_3)
        b.bn['a', 'd', 4, 5, 'a', 1].v2 = Var(m.d1_1, m.d1_2)
        b.bn['a', 'd', 4, 5, 'a', 1].vn = Var(m.d1_1, m.dn, m.d2)
    return m