import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
def test_otdered_sorted_iter(self):
    m = ConcreteModel()

    @m.Block([2, 1], [4, 5])
    def b(b, i, j):
        b.x = Var([8, 7], initialize=0)
    rs = _ReferenceSet(m.b[...].x[:])
    self.assertEqual(list(rs), [(2, 4, 8), (2, 4, 7), (2, 5, 8), (2, 5, 7), (1, 4, 8), (1, 4, 7), (1, 5, 8), (1, 5, 7)])
    rs = _ReferenceSet(m.b[...].x[:])
    self.assertEqual(list(rs.ordered_iter()), [(2, 4, 8), (2, 4, 7), (2, 5, 8), (2, 5, 7), (1, 4, 8), (1, 4, 7), (1, 5, 8), (1, 5, 7)])
    rs = _ReferenceSet(m.b[...].x[:])
    self.assertEqual(list(rs.sorted_iter()), [(1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8), (2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8)])
    m = ConcreteModel()
    m.I = FiniteSetOf([2, 1])
    m.J = FiniteSetOf([4, 5])
    m.K = FiniteSetOf([8, 7])

    @m.Block(m.I, m.J)
    def b(b, i, j):
        b.x = Var(m.K, initialize=0)
    rs = _ReferenceSet(m.b[...].x[:])
    self.assertEqual(list(rs), [(2, 4, 8), (2, 4, 7), (2, 5, 8), (2, 5, 7), (1, 4, 8), (1, 4, 7), (1, 5, 8), (1, 5, 7)])
    rs = _ReferenceSet(m.b[...].x[:])
    self.assertEqual(list(rs.ordered_iter()), [(1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8), (2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8)])
    rs = _ReferenceSet(m.b[...].x[:])
    self.assertEqual(list(rs.sorted_iter()), [(1, 4, 7), (1, 4, 8), (1, 5, 7), (1, 5, 8), (2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8)])