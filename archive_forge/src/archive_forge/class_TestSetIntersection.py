import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
class TestSetIntersection(unittest.TestCase):

    def test_pickle(self):
        a = SetOf([1, 3, 5]) & SetOf([2, 3, 4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)

    def test_bounds(self):
        a = SetOf([-2, -1, 0, 1])
        b = a & NonNegativeReals
        self.assertEqual(b.bounds(), (0, 1))
        b = NonNegativeReals & a
        self.assertEqual(b.bounds(), (0, 1))
        b = a & RangeSet(3)
        self.assertEqual(b.bounds(), (1, 1))

    def test_naming(self):
        m = ConcreteModel()
        m.I = SetOf([1, 2])
        a = m.I & [3, 4]
        b = [-1, 1] & a
        self.assertEqual(str(a), 'I & {3, 4}')
        self.assertEqual(str(b), '{-1, 1} & (I & {3, 4})')
        m.A = a
        self.assertEqual(str(a), 'A')
        self.assertEqual(str(b), '{-1, 1} & A')

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1, 2])
        m.A = m.I & [3, 4]
        self.assertIs(m.A._domain, m.A)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegex(ValueError, 'Setting the domain of a Set Operator is not allowed'):
            m.A._domain = None
        output = StringIO()
        m.A.pprint(ostream=output)
        ref = '\nA : Size=1, Index=None, Ordered=True\n    Key  : Dimen : Domain     : Size : Members\n    None :     1 : I & {3, 4} :    0 :      {}\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1, 2, 3, 4])
        m.I2 = SetOf([(1, 2), (3, 4)])
        m.IN = SetOf([(1, 2), (3, 4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 & m.I1).dimen, 1)
        self.assertEqual((m.I2 & m.I2).dimen, 2)
        self.assertEqual((m.IN & m.IN).dimen, None)
        self.assertEqual((m.I1 & m.I2).dimen, 0)
        self.assertEqual((m.IN & m.I2).dimen, 2)
        self.assertEqual((m.I2 & m.IN).dimen, 2)
        self.assertEqual((m.IN & m.I1).dimen, 1)
        self.assertEqual((m.I1 & m.IN).dimen, 1)
        self.assertEqual((m.I1 & m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.I2 & m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.IN & m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.J & m.I1).dimen, UnknownSetDimen)
        self.assertEqual((m.J & m.I2).dimen, UnknownSetDimen)
        self.assertEqual((m.J & m.IN).dimen, UnknownSetDimen)

    def _verify_ordered_intersection(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(a_ordered or b_ordered)
        if a_ordered:
            ref = (3, 2, 5)
        else:
            ref = (2, 3, 5)
        x = a & b
        self.assertIs(type(x), SetIntersection_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
        self.assertEqual(len(x), 3)
        self.assertEqual(list(x), list(ref))
        self.assertEqual(x.ordered_data(), tuple(ref))
        self.assertEqual(x.sorted_data(), (2, 3, 5))
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)
        self.assertEqual(x.ord(2), ref.index(2) + 1)
        self.assertEqual(x.ord(3), ref.index(3) + 1)
        self.assertEqual(x.ord(5), 3)
        with self.assertRaisesRegex(IndexError, 'Cannot identify position of 6 in Set SetIntersection_OrderedSet'):
            x.ord(6)
        self.assertEqual(x[1], ref[0])
        self.assertEqual(x[2], ref[1])
        self.assertEqual(x[3], 5)
        with self.assertRaisesRegex(IndexError, 'SetIntersection_OrderedSet index out of range'):
            x[4]
        self.assertEqual(x[-1], 5)
        self.assertEqual(x[-2], ref[-2])
        self.assertEqual(x[-3], ref[-3])
        with self.assertRaisesRegex(IndexError, 'SetIntersection_OrderedSet index out of range'):
            x[-4]

    def test_ordered_setintersection(self):
        self._verify_ordered_intersection(SetOf([1, 3, 2, 5]), SetOf([0, 2, 3, 4, 5]))
        self._verify_ordered_intersection(SetOf([1, 3, 2, 5]), SetOf({0, 2, 3, 4, 5}))
        self._verify_ordered_intersection(SetOf({1, 3, 2, 5}), SetOf([0, 2, 3, 4, 5]))
        self._verify_ordered_intersection(SetOf([1, 3, 2, 5]), [0, 2, 3, 4, 5])
        self._verify_ordered_intersection(SetOf([1, 3, 2, 5]), {0, 2, 3, 4, 5})
        self._verify_ordered_intersection([1, 3, 2, 5], SetOf([0, 2, 3, 4, 5]))
        self._verify_ordered_intersection({1, 3, 2, 5}, SetOf([0, 2, 3, 4, 5]))

    def _verify_finite_intersection(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_finite = a.isfinite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.isfinite()
        else:
            b_finite = True
        self.assertTrue(a_finite or b_finite)
        x = a & b
        self.assertIs(type(x), SetIntersection_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertEqual(len(x), 3)
        if x._sets[0].isordered():
            self.assertEqual(list(x)[:3], [3, 2, 5])
        self.assertEqual(sorted(list(x)), [2, 3, 5])
        self.assertEqual(x.ordered_data(), (2, 3, 5))
        self.assertEqual(x.sorted_data(), (2, 3, 5))
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertNotIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)
        self.assertEqual(len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 9)
        self.assertEqual(len(list(x.ranges())), 3)

    def test_finite_setintersection(self):
        self._verify_finite_intersection(SetOf({1, 3, 2, 5}), SetOf({0, 2, 3, 4, 5}))
        self._verify_finite_intersection({1, 3, 2, 5}, SetOf({0, 2, 3, 4, 5}))
        self._verify_finite_intersection(SetOf({1, 3, 2, 5}), {0, 2, 3, 4, 5})
        self._verify_finite_intersection(RangeSet(ranges=(NR(-5, -1, 0), NR(2, 3, 0), NR(5, 5, 0), NR(10, 20, 0))), SetOf({0, 2, 3, 4, 5}))
        self._verify_finite_intersection(SetOf({1, 3, 2, 5}), RangeSet(ranges=(NR(2, 5, 0), NR(2, 5, 0), NR(6, 6, 0), NR(6, 6, 0), NR(6, 6, 0))))

    def _verify_infinite_intersection(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_finite = a.isfinite()
        else:
            a_finite = True
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_finite = b.isfinite()
        else:
            b_finite = True
        self.assertEqual([a_finite, b_finite], [False, False])
        x = a & b
        self.assertIs(type(x), SetIntersection_InfiniteSet)
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertNotIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertIn(4, x)
        self.assertNotIn(5, x)
        self.assertNotIn(6, x)
        self.assertEqual(list(x.ranges()), list(RangeSet(2, 4, 0).ranges()))

    def test_infinite_setintersection(self):
        self._verify_infinite_intersection(RangeSet(0, 4, 0), RangeSet(2, 6, 0))

    def test_odd_intersections(self):
        m = AbstractModel()
        m.p = Param(initialize=0)
        m.a = RangeSet(0, None, 2)
        m.b = RangeSet(5, 10, m.p, finite=False)
        m.x = m.a & m.b
        self.assertTrue(m.a._constructed)
        self.assertFalse(m.b._constructed)
        self.assertFalse(m.x._constructed)
        self.assertIs(type(m.x), SetIntersection_InfiniteSet)
        i = m.create_instance()
        self.assertIs(type(i.x), SetIntersection_OrderedSet)
        self.assertEqual(list(i.x), [6, 8, 10])
        self.assertEqual(i.x.ord(6), 1)
        self.assertEqual(i.x.ord(8), 2)
        self.assertEqual(i.x.ord(10), 3)
        self.assertEqual(i.x[1], 6)
        self.assertEqual(i.x[2], 8)
        self.assertEqual(i.x[3], 10)
        with self.assertRaisesRegex(IndexError, 'x index out of range'):
            i.x[4]
        self.assertEqual(i.x[-3], 6)
        self.assertEqual(i.x[-2], 8)
        self.assertEqual(i.x[-1], 10)
        with self.assertRaisesRegex(IndexError, 'x index out of range'):
            i.x[-4]

    def test_subsets(self):
        a = SetOf([1])
        b = SetOf([1])
        c = SetOf([1])
        d = SetOf([1])
        x = a & b
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [x])
        self.assertEqual(list(x.subsets(False)), [x])
        self.assertEqual(list(x.subsets(True)), [a, b])
        x = a & b & c
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [x])
        self.assertEqual(list(x.subsets(False)), [x])
        self.assertEqual(list(x.subsets(True)), [a, b, c])
        x = a & b & (c & d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [x])
        self.assertEqual(list(x.subsets(False)), [x])
        self.assertEqual(list(x.subsets(True)), [a, b, c, d])
        x = (a & b) * (c & d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(len(list(x.subsets())), 2)
        self.assertEqual(list(x.subsets()), [a & b, c & d])
        self.assertEqual(list(x.subsets(False)), [a & b, c & d])
        self.assertEqual(len(list(x.subsets(True))), 4)
        self.assertEqual(list(x.subsets(True)), [a, b, c, d])