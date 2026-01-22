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
class TestReferenceSet(unittest.TestCase):

    def test_str(self):
        m = ConcreteModel()

        @m.Block([1, 2], [4, 5])
        def b(b, i, j):
            b.x = Var([7, 8], [10, 11], initialize=0)
            b.y = Var([7, 8], initialize=0)
            b.z = Var()
        rs = _ReferenceSet(m.b[:, 5].z)
        self.assertEqual(str(rs), 'ReferenceSet(b[:, 5].z)')

    def test_lookup_and_iter_dense_data(self):
        m = ConcreteModel()

        @m.Block([1, 2], [4, 5])
        def b(b, i, j):
            b.x = Var([7, 8], [10, 11], initialize=0)
            b.y = Var([7, 8], initialize=0)
            b.z = Var()
        rs = _ReferenceSet(m.b[:, 5].z)
        self.assertNotIn((0,), rs)
        self.assertIn(1, rs)
        self.assertIn((1,), rs)
        self.assertEqual(len(rs), 2)
        self.assertEqual(list(rs), [1, 2])
        rs = _ReferenceSet(m.b[:, 5].bad)
        self.assertNotIn((0,), rs)
        self.assertNotIn((1,), rs)
        self.assertEqual(len(rs), 0)
        self.assertEqual(list(rs), [])

        @m.Block([1, 2, 3])
        def d(b, i):
            if i % 2:
                b.x = Var(range(i))
        rs = _ReferenceSet(m.d[:].x[:])
        self.assertIn((1, 0), rs)
        self.assertIn((3, 0), rs)
        self.assertNotIn((2, 0), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(1, 0), (3, 0), (3, 1), (3, 2)])
        rs = _ReferenceSet(m.d[...].x[...])
        self.assertIn((1, 0), rs)
        self.assertIn((3, 0), rs)
        self.assertNotIn((2, 0), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(1, 0), (3, 0), (3, 1), (3, 2)])
        m.e_index = Set(initialize=[2, (2, 3)], dimen=None)

        @m.Block(m.e_index)
        def e(b, *args):
            b.x_index = Set(initialize=[1, (3, 4)], dimen=None)
            b.x = Var(b.x_index)
        rs = _ReferenceSet(m.e[...].x[...])
        self.assertIn((2, 1), rs)
        self.assertIn((2, 3, 1), rs)
        self.assertIn((2, 3, 4), rs)
        self.assertNotIn((2, 3, 5), rs)
        self.assertEqual(len(rs), 4)
        self.assertEqual(list(rs), [(2, 1), (2, 3, 4), (2, 3, 1), (2, 3, 3, 4)])
        rs = _ReferenceSet(m.e[...])
        self.assertIn(2, rs)
        self.assertIn((2,), rs)
        self.assertNotIn(0, rs)
        self.assertNotIn((0,), rs)

    def test_lookup_and_iter_sparse_data(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.x = Var(m.I, m.I, dense=False)
        rd = _ReferenceDict(m.x[...])
        rs = _ReferenceSet(m.x[...])
        self.assertEqual(len(rd), 0)
        self.assertEqual(len(rd), 0)
        self.assertEqual(len(rs), 9)
        self.assertEqual(len(rd), 0)
        self.assertIn((1, 1), rs)
        self.assertEqual(len(rd), 0)
        self.assertEqual(len(rs), 9)

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