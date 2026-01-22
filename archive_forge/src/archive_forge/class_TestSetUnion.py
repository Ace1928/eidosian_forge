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
class TestSetUnion(unittest.TestCase):

    def test_pickle(self):
        a = SetOf([1, 3, 5]) | SetOf([2, 3, 4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)

    def test_len(self):
        a = SetOf([1, 2, 3])
        self.assertEqual(len(a), 3)
        b = a | Reals
        with self.assertRaisesRegex(OverflowError, 'The length of a non-finite Set is Inf'):
            len(b)

    def test_bounds(self):
        a = SetOf([-2, -1, 0, 1])
        b = a | NonNegativeReals
        self.assertEqual(b.bounds(), (-2, None))
        b = NonNegativeReals | a
        self.assertEqual(b.bounds(), (-2, None))
        b = a | RangeSet(3)
        self.assertEqual(b.bounds(), (-2, 3))
        b = NegativeReals | NonNegativeReals
        self.assertEqual(b.bounds(), (None, None))

    def test_naming(self):
        m = ConcreteModel()
        m.I = SetOf([1, 2])
        a = m.I | [3, 4]
        b = [-1, 1] | a
        self.assertEqual(str(a), 'I | {3, 4}')
        self.assertEqual(str(b), '{-1, 1} | (I | {3, 4})')
        m.A = a
        self.assertEqual(str(a), 'A')
        self.assertEqual(str(b), '{-1, 1} | A')

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1, 2])
        m.A = m.I | [3, 4]
        self.assertIs(m.A._domain, m.A)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegex(ValueError, 'Setting the domain of a Set Operator is not allowed'):
            m.A._domain = None
        output = StringIO()
        m.A.pprint(ostream=output)
        ref = '\nA : Size=1, Index=None, Ordered=True\n    Key  : Dimen : Domain     : Size : Members\n    None :     1 : I | {3, 4} :    4 : {1, 2, 3, 4}\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1, 2, 3, 4])
        m.I2 = SetOf([(1, 2), (3, 4)])
        m.IN = SetOf([(1, 2), (3, 4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 | m.I1).dimen, 1)
        self.assertEqual((m.I2 | m.I2).dimen, 2)
        self.assertEqual((m.IN | m.IN).dimen, None)
        self.assertEqual((m.I1 | m.I2).dimen, None)
        self.assertEqual((m.IN | m.I2).dimen, None)
        self.assertEqual((m.I2 | m.IN).dimen, None)
        self.assertEqual((m.IN | m.I1).dimen, None)
        self.assertEqual((m.I1 | m.IN).dimen, None)
        self.assertEqual((m.I1 | m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.I2 | m.J).dimen, UnknownSetDimen)
        self.assertEqual((m.IN | m.J).dimen, None)
        self.assertEqual((m.J | m.I1).dimen, UnknownSetDimen)
        self.assertEqual((m.J | m.I2).dimen, UnknownSetDimen)
        self.assertEqual((m.J | m.IN).dimen, None)

    def _verify_ordered_union(self, a, b):
        if isinstance(a, SetOf):
            self.assertTrue(a.isordered())
            self.assertTrue(a.isfinite())
        else:
            self.assertIs(type(a), list)
        if isinstance(b, SetOf):
            self.assertTrue(b.isordered())
            self.assertTrue(b.isfinite())
        else:
            self.assertIs(type(b), list)
        x = a | b
        self.assertIs(type(x), SetUnion_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
        self.assertEqual(len(x), 5)
        self.assertEqual(list(x), [1, 3, 2, 5, 4])
        self.assertEqual(x.ordered_data(), (1, 3, 2, 5, 4))
        self.assertEqual(x.sorted_data(), (1, 2, 3, 4, 5))
        self.assertIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)
        self.assertEqual(x.ord(1), 1)
        self.assertEqual(x.ord(2), 3)
        self.assertEqual(x.ord(3), 2)
        self.assertEqual(x.ord(4), 5)
        self.assertEqual(x.ord(5), 4)
        with self.assertRaisesRegex(IndexError, 'Cannot identify position of 6 in Set SetUnion_OrderedSet'):
            x.ord(6)
        self.assertEqual(x[1], 1)
        self.assertEqual(x[2], 3)
        self.assertEqual(x[3], 2)
        self.assertEqual(x[4], 5)
        self.assertEqual(x[5], 4)
        with self.assertRaisesRegex(IndexError, 'SetUnion_OrderedSet index out of range'):
            x[6]
        self.assertEqual(x[-1], 4)
        self.assertEqual(x[-2], 5)
        self.assertEqual(x[-3], 2)
        self.assertEqual(x[-4], 3)
        self.assertEqual(x[-5], 1)
        with self.assertRaisesRegex(IndexError, 'SetUnion_OrderedSet index out of range'):
            x[-6]

    def test_ordered_setunion(self):
        self._verify_ordered_union(SetOf([1, 3, 2]), SetOf([5, 3, 4]))
        self._verify_ordered_union([1, 3, 2], SetOf([5, 3, 4]))
        self._verify_ordered_union(SetOf([1, 3, 2]), [5, 3, 4])

    def _verify_finite_union(self, a, b):
        if isinstance(a, SetOf):
            if type(a._ref) is list:
                self.assertTrue(a.isordered())
            else:
                self.assertFalse(a.isordered())
            self.assertTrue(a.isfinite())
        else:
            self.assertIn(type(a), (list, set))
        if isinstance(b, SetOf):
            if type(b._ref) is list:
                self.assertTrue(b.isordered())
            else:
                self.assertFalse(b.isordered())
            self.assertTrue(b.isfinite())
        else:
            self.assertIn(type(b), (list, set))
        x = a | b
        self.assertIs(type(x), SetUnion_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertEqual(len(x), 5)
        if x._sets[0].isordered():
            self.assertEqual(list(x)[:3], [1, 3, 2])
        if x._sets[1].isordered():
            self.assertEqual(list(x)[-2:], [5, 4])
        self.assertEqual(sorted(list(x)), [1, 2, 3, 4, 5])
        self.assertEqual(x.ordered_data(), (1, 2, 3, 4, 5))
        self.assertEqual(x.sorted_data(), (1, 2, 3, 4, 5))
        self.assertIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)
        self.assertEqual(len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 6)
        self.assertEqual(len(list(x.ranges())), 5)

    def test_finite_setunion(self):
        self._verify_finite_union(SetOf({1, 3, 2}), SetOf({5, 3, 4}))
        self._verify_finite_union([1, 3, 2], SetOf({5, 3, 4}))
        self._verify_finite_union(SetOf({1, 3, 2}), [5, 3, 4])
        self._verify_finite_union({1, 3, 2}, SetOf([5, 3, 4]))
        self._verify_finite_union(SetOf([1, 3, 2]), {5, 3, 4})

    def _verify_infinite_union(self, a, b):
        if isinstance(a, RangeSet):
            self.assertFalse(a.isordered())
            self.assertFalse(a.isfinite())
        else:
            self.assertIn(type(a), (list, set))
        if isinstance(b, RangeSet):
            self.assertFalse(b.isordered())
            self.assertFalse(b.isfinite())
        else:
            self.assertIn(type(b), (list, set))
        x = a | b
        self.assertIs(type(x), SetUnion_InfiniteSet)
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertIn(1, x)
        self.assertIn(2, x)
        self.assertIn(3, x)
        self.assertIn(4, x)
        self.assertIn(5, x)
        self.assertNotIn(6, x)
        self.assertEqual(list(x.ranges()), list(x._sets[0].ranges()) + list(x._sets[1].ranges()))

    def test_infinite_setunion(self):
        self._verify_infinite_union(RangeSet(1, 3, 0), RangeSet(3, 5, 0))
        self._verify_infinite_union([1, 3, 2], RangeSet(3, 5, 0))
        self._verify_infinite_union(RangeSet(1, 3, 0), [5, 3, 4])
        self._verify_infinite_union({1, 3, 2}, RangeSet(3, 5, 0))
        self._verify_infinite_union(RangeSet(1, 3, 0), {5, 3, 4})

    def test_invalid_operators(self):
        m = ConcreteModel()
        m.I = RangeSet(5)
        m.J = Set([1, 2])
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(J\\)'):
            m.I | m.J
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(J\\)'):
            m.J | m.I
        m.x = Suffix()
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to a non-Set Suffix component \\(x\\)'):
            m.I | m.x
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to a non-Set Suffix component \\(x\\)'):
            m.x | m.I
        m.y = Var([1, 2])
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Var component \\(y\\)'):
            m.I | m.y
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to a non-Set component data \\(y\\[1\\]\\)'):
            m.I | m.y[1]
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Var component \\(y\\)'):
            m.y | m.I
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to a non-Set component data \\(y\\[1\\]\\)'):
            m.y[1] | m.I