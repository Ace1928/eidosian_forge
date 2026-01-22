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
class TestReferenceDict(unittest.TestCase):

    def setUp(self):
        self.m = m = ConcreteModel()

        @m.Block([1, 2], [4, 5])
        def b(b, i, j):
            b.x = Var([7, 8], [10, 11], initialize=0)
            b.y = Var([7, 8], initialize=0)
            b.z = Var()

        @m.Block([1, 2])
        def c(b, i):
            b.x = Var([7, 8], [10, 11], initialize=0)
            b.y = Var([7, 8], initialize=0)
            b.z = Var()

    def _lookupTester(self, _slice, key, ans):
        rd = _ReferenceDict(_slice)
        self.assertIn(key, rd)
        self.assertIs(rd[key], ans)
        if len(key) == 1:
            self.assertIn(key[0], rd)
            self.assertIs(rd[key[0]], ans)
        self.assertNotIn(None, rd)
        with self.assertRaises(KeyError):
            rd[None]
        for i in range(len(key)):
            _ = tuple([0] * i)
            self.assertNotIn(_, rd)
            with self.assertRaises(KeyError):
                rd[_]

    def test_simple_lookup(self):
        m = self.m
        self._lookupTester(m.b[:, :].x[:, :], (1, 5, 7, 10), m.b[1, 5].x[7, 10])
        self._lookupTester(m.b[:, 4].x[8, :], (1, 10), m.b[1, 4].x[8, 10])
        self._lookupTester(m.b[:, 4].x[8, 10], (1,), m.b[1, 4].x[8, 10])
        self._lookupTester(m.b[1, 4].x[8, :], (10,), m.b[1, 4].x[8, 10])
        self._lookupTester(m.b[:, :].y[:], (1, 5, 7), m.b[1, 5].y[7])
        self._lookupTester(m.b[:, 4].y[:], (1, 7), m.b[1, 4].y[7])
        self._lookupTester(m.b[:, 4].y[8], (1,), m.b[1, 4].y[8])
        self._lookupTester(m.b[:, :].z, (1, 5), m.b[1, 5].z)
        self._lookupTester(m.b[:, 4].z, (1,), m.b[1, 4].z)
        self._lookupTester(m.c[:].x[:, :], (1, 7, 10), m.c[1].x[7, 10])
        self._lookupTester(m.c[:].x[8, :], (1, 10), m.c[1].x[8, 10])
        self._lookupTester(m.c[:].x[8, 10], (1,), m.c[1].x[8, 10])
        self._lookupTester(m.c[1].x[:, :], (8, 10), m.c[1].x[8, 10])
        self._lookupTester(m.c[1].x[8, :], (10,), m.c[1].x[8, 10])
        self._lookupTester(m.c[:].y[:], (1, 7), m.c[1].y[7])
        self._lookupTester(m.c[:].y[8], (1,), m.c[1].y[8])
        self._lookupTester(m.c[1].y[:], (8,), m.c[1].y[8])
        self._lookupTester(m.c[:].z, (1,), m.c[1].z)
        m.jagged_set = Set(initialize=[1, (2, 3)], dimen=None)
        m.jb = Block(m.jagged_set)
        m.jb[1].x = Var([1, 2, 3])
        m.jb[2, 3].x = Var([1, 2, 3])
        self._lookupTester(m.jb[...], (1,), m.jb[1])
        self._lookupTester(m.jb[...].x[:], (1, 2), m.jb[1].x[2])
        self._lookupTester(m.jb[...].x[:], (2, 3, 2), m.jb[2, 3].x[2])
        rd = _ReferenceDict(m.jb[:, :, :].x[:])
        with self.assertRaises(KeyError):
            rd[2, 3, 4, 2]
        rd = _ReferenceDict(m.b[:, 4].x[:])
        with self.assertRaises(KeyError):
            rd[1, 0]

    def test_len(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, :].x[:, :])
        self.assertEqual(len(rd), 2 * 2 * 2 * 2)
        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(len(rd), 2 * 2)

    def test_iterators(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(list(rd), [(1, 10), (1, 11), (2, 10), (2, 11)])
        self.assertEqual(list(rd.keys()), [(1, 10), (1, 11), (2, 10), (2, 11)])
        self.assertEqual(list(rd.values()), [m.b[1, 4].x[8, 10], m.b[1, 4].x[8, 11], m.b[2, 4].x[8, 10], m.b[2, 4].x[8, 11]])
        self.assertEqual(list(rd.items()), [((1, 10), m.b[1, 4].x[8, 10]), ((1, 11), m.b[1, 4].x[8, 11]), ((2, 10), m.b[2, 4].x[8, 10]), ((2, 11), m.b[2, 4].x[8, 11])])

    def test_ordered_iterators(self):
        m = ConcreteModel()
        m.I = Set(initialize=[3, 2])
        m.b = Block([1, 0])
        m.b[1].x = Var(m.I)
        m.b[0].x = Var(m.I)
        m.y = Reference(m.b[:].x[:])
        self.assertEqual(list(m.y.index_set().subsets()), [m.b.index_set(), m.I])
        self.assertEqual(list(m.y), [(1, 3), (1, 2), (0, 3), (0, 2)])
        self.assertEqual(list(m.y.keys()), [(1, 3), (1, 2), (0, 3), (0, 2)])
        self.assertEqual(list(m.y.values()), [m.b[1].x[3], m.b[1].x[2], m.b[0].x[3], m.b[0].x[2]])
        self.assertEqual(list(m.y.items()), [((1, 3), m.b[1].x[3]), ((1, 2), m.b[1].x[2]), ((0, 3), m.b[0].x[3]), ((0, 2), m.b[0].x[2])])
        self.assertEqual(list(m.y.keys(True)), [(0, 2), (0, 3), (1, 2), (1, 3)])
        self.assertEqual(list(m.y.values(True)), [m.b[0].x[2], m.b[0].x[3], m.b[1].x[2], m.b[1].x[3]])
        self.assertEqual(list(m.y.items(True)), [((0, 2), m.b[0].x[2]), ((0, 3), m.b[0].x[3]), ((1, 2), m.b[1].x[2]), ((1, 3), m.b[1].x[3])])
        m = ConcreteModel()
        m.b = Block([1, 0])
        m.b[1].x = Var([3, 2])
        m.b[0].x = Var([5, 4])
        m.y = Reference(m.b[:].x[:])
        self.assertIs(type(m.y.index_set()), FiniteSetOf)
        self.assertEqual(list(m.y), [(1, 3), (1, 2), (0, 5), (0, 4)])
        self.assertEqual(list(m.y.keys()), [(1, 3), (1, 2), (0, 5), (0, 4)])
        self.assertEqual(list(m.y.values()), [m.b[1].x[3], m.b[1].x[2], m.b[0].x[5], m.b[0].x[4]])
        self.assertEqual(list(m.y.items()), [((1, 3), m.b[1].x[3]), ((1, 2), m.b[1].x[2]), ((0, 5), m.b[0].x[5]), ((0, 4), m.b[0].x[4])])
        self.assertEqual(list(m.y.keys(True)), [(0, 4), (0, 5), (1, 2), (1, 3)])
        self.assertEqual(list(m.y.values(True)), [m.b[0].x[4], m.b[0].x[5], m.b[1].x[2], m.b[1].x[3]])
        self.assertEqual(list(m.y.items(True)), [((0, 4), m.b[0].x[4]), ((0, 5), m.b[0].x[5]), ((1, 2), m.b[1].x[2]), ((1, 3), m.b[1].x[3])])
        m = ConcreteModel()
        m.b = Block([1, 0])
        m.b[1].x = Var([3, 2])
        m.b[0].x = Var([5, 4])
        m.y = Reference({(1, 3): m.b[1].x[3], (0, 5): m.b[0].x[5], (1, 2): m.b[1].x[2], (0, 4): m.b[0].x[4]})
        self.assertIs(type(m.y.index_set()), FiniteSetOf)
        self.assertEqual(list(m.y), [(1, 3), (0, 5), (1, 2), (0, 4)])
        self.assertEqual(list(m.y.keys()), [(1, 3), (0, 5), (1, 2), (0, 4)])
        self.assertEqual(list(m.y.values()), [m.b[1].x[3], m.b[0].x[5], m.b[1].x[2], m.b[0].x[4]])
        self.assertEqual(list(m.y.items()), [((1, 3), m.b[1].x[3]), ((0, 5), m.b[0].x[5]), ((1, 2), m.b[1].x[2]), ((0, 4), m.b[0].x[4])])
        self.assertEqual(list(m.y.keys(True)), [(0, 4), (0, 5), (1, 2), (1, 3)])
        self.assertEqual(list(m.y.values(True)), [m.b[0].x[4], m.b[0].x[5], m.b[1].x[2], m.b[1].x[3]])
        self.assertEqual(list(m.y.items(True)), [((0, 4), m.b[0].x[4]), ((0, 5), m.b[0].x[5]), ((1, 2), m.b[1].x[2]), ((1, 3), m.b[1].x[3])])

    def test_nested_assignment(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, :].x[:, :])
        self.assertEqual(sum((x.value for x in rd.values())), 0)
        rd[1, 5, 7, 10] = 10
        self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
        self.assertEqual(sum((x.value for x in rd.values())), 10)
        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(sum((x.value for x in rd.values())), 0)
        rd[1, 10] = 20
        self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
        self.assertEqual(sum((x.value for x in rd.values())), 20)

    def test_attribute_assignment(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, :].x[:, :].value)
        self.assertEqual(sum((x for x in rd.values())), 0)
        rd[1, 5, 7, 10] = 10
        self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
        self.assertEqual(sum((x for x in rd.values())), 10)
        rd = _ReferenceDict(m.b[:, 4].x[8, :].value)
        self.assertEqual(sum((x for x in rd.values())), 0)
        rd[1, 10] = 20
        self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
        self.assertEqual(sum((x for x in rd.values())), 20)
        m.x = Var([1, 2], initialize=0)
        rd = _ReferenceDict(m.x[:])
        self.assertEqual(sum((x.value for x in rd.values())), 0)
        rd[2] = 10
        self.assertEqual(m.x[1].value, 0)
        self.assertEqual(m.x[2].value, 10)
        self.assertEqual(sum((x.value for x in rd.values())), 10)

    def test_single_attribute_assignment(self):
        m = self.m
        rd = _ReferenceDict(m.b[1, 5].x[:, :])
        self.assertEqual(sum((x.value for x in rd.values())), 0)
        rd[7, 10].value = 10
        self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
        self.assertEqual(sum((x.value for x in rd.values())), 10)
        rd = _ReferenceDict(m.b[1, 4].x[8, :])
        self.assertEqual(sum((x.value for x in rd.values())), 0)
        rd[10].value = 20
        self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
        self.assertEqual(sum((x.value for x in rd.values())), 20)

    def test_nested_attribute_assignment(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, :].x[:, :])
        self.assertEqual(sum((x.value for x in rd.values())), 0)
        rd[1, 5, 7, 10].value = 10
        self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
        self.assertEqual(sum((x.value for x in rd.values())), 10)
        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(sum((x.value for x in rd.values())), 0)
        rd[1, 10].value = 20
        self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
        self.assertEqual(sum((x.value for x in rd.values())), 20)

    def test_single_deletion(self):
        m = self.m
        rd = _ReferenceDict(m.b[1, 5].x[:, :])
        self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2)
        self.assertTrue((7, 10) in rd)
        del rd[7, 10]
        self.assertFalse((7, 10) in rd)
        self.assertEqual(len(list((x.value for x in rd.values()))), 3)
        rd = _ReferenceDict(m.b[1, 4].x[8, :])
        self.assertEqual(len(list((x.value for x in rd.values()))), 2)
        self.assertTrue(10 in rd)
        del rd[10]
        self.assertFalse(10 in rd)
        self.assertEqual(len(list((x.value for x in rd.values()))), 2 - 1)
        with self.assertRaisesRegex(KeyError, "\\(8, 10\\) is not valid for indexed component 'b\\[1,4\\].x'"):
            del rd[10]
        rd = _ReferenceDict(m.b[1, :].x[8, 0])
        with self.assertRaisesRegex(KeyError, "'\\(8, 0\\)' is not valid for indexed component 'b\\[1,4\\].x'"):
            del rd[4]

    def test_nested_deletion(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, :].x[:, :])
        self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2 * 2 * 2)
        self.assertTrue((1, 5, 7, 10) in rd)
        del rd[1, 5, 7, 10]
        self.assertFalse((1, 5, 7, 10) in rd)
        self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2 * 2 * 2 - 1)
        rd = _ReferenceDict(m.b[:, 4].x[8, :])
        self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2)
        self.assertTrue((1, 10) in rd)
        del rd[1, 10]
        self.assertFalse((1, 10) in rd)
        self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2 - 1)

    def test_attribute_deletion(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, :].z)
        rd._slice.attribute_errors_generate_exceptions = False
        self.assertEqual(len(list((x.value for x in rd.values()))), 2 * 2)
        self.assertTrue((1, 5) in rd)
        self.assertTrue(hasattr(m.b[1, 5], 'z'))
        self.assertTrue(hasattr(m.b[2, 5], 'z'))
        del rd[1, 5]
        self.assertFalse((1, 5) in rd)
        self.assertFalse(hasattr(m.b[1, 5], 'z'))
        self.assertTrue(hasattr(m.b[2, 5], 'z'))
        self.assertEqual(len(list((x.value for x in rd.values()))), 3)
        rd = _ReferenceDict(m.b[2, :].z)
        rd._slice.attribute_errors_generate_exceptions = False
        self.assertEqual(len(list((x.value for x in rd.values()))), 2)
        self.assertTrue(5 in rd)
        self.assertTrue(hasattr(m.b[2, 4], 'z'))
        self.assertTrue(hasattr(m.b[2, 5], 'z'))
        del rd[5]
        self.assertFalse(5 in rd)
        self.assertTrue(hasattr(m.b[2, 4], 'z'))
        self.assertFalse(hasattr(m.b[2, 5], 'z'))
        self.assertEqual(len(list((x.value for x in rd.values()))), 2 - 1)

    def test_deprecations(self):
        m = self.m
        rd = _ReferenceDict(m.b[:, :].z)
        items = rd.items()
        with LoggingIntercept() as LOG:
            iteritems = rd.iteritems()
        self.assertIs(type(items), type(iteritems))
        self.assertEqual(list(items), list(iteritems))
        self.assertIn('DEPRECATED: The iteritems method is deprecated. Use dict.items', LOG.getvalue())
        values = rd.values()
        with LoggingIntercept() as LOG:
            itervalues = rd.itervalues()
        self.assertIs(type(values), type(itervalues))
        self.assertEqual(list(values), list(itervalues))
        self.assertIn('DEPRECATED: The itervalues method is deprecated. Use dict.values', LOG.getvalue())