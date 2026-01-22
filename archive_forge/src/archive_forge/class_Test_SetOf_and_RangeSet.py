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
class Test_SetOf_and_RangeSet(unittest.TestCase):

    def test_constructor(self):
        i = SetOf([1, 2, 3])
        self.assertIs(type(i), OrderedSetOf)
        j = OrderedSetOf([1, 2, 3])
        self.assertIs(type(i), OrderedSetOf)
        self.assertEqual(i, j)
        i = SetOf({1, 2, 3})
        self.assertIs(type(i), FiniteSetOf)
        j = FiniteSetOf([1, 2, 3])
        self.assertIs(type(i), FiniteSetOf)
        self.assertEqual(i, j)
        i = SetOf(NonNegativeReals)
        self.assertIs(type(i), InfiniteSetOf)
        j = InfiniteSetOf(NonNegativeReals)
        self.assertIs(type(i), InfiniteSetOf)
        self.assertEqual(i, j)
        i = SetOf(Binary)
        self.assertIs(type(i), OrderedSetOf)
        j = OrderedSetOf(Binary)
        self.assertIs(type(i), OrderedSetOf)
        self.assertEqual(i, j)
        I = Set(initialize={1, 3, 2}, ordered=False)
        I.construct()
        i = SetOf(I)
        self.assertIs(type(i), FiniteSetOf)
        j = FiniteSetOf(I)
        self.assertIs(type(i), FiniteSetOf)
        self.assertEqual(i, j)
        i = RangeSet(3)
        self.assertTrue(i.is_constructed())
        self.assertEqual(len(i), 3)
        self.assertEqual(len(list(i.ranges())), 1)
        i = RangeSet(1, 3)
        self.assertTrue(i.is_constructed())
        self.assertEqual(len(i), 3)
        self.assertEqual(len(list(i.ranges())), 1)
        i = RangeSet(ranges=[NR(1, 3, 1)])
        self.assertTrue(i.is_constructed())
        self.assertEqual(len(i), 3)
        self.assertEqual(list(i.ranges()), [NR(1, 3, 1)])
        i = RangeSet(1, 3, 0)
        with self.assertRaisesRegex(TypeError, ".*'InfiniteScalarRangeSet' has no len"):
            len(i)
        self.assertEqual(len(list(i.ranges())), 1)
        with self.assertRaisesRegex(TypeError, ".*'GlobalSet' has no len"):
            len(Integers)
        self.assertEqual(len(list(Integers.ranges())), 2)
        with self.assertRaisesRegex(ValueError, 'RangeSet expects 3 or fewer positional arguments \\(received 4\\)'):
            RangeSet(1, 2, 3, 4)
        with self.assertRaisesRegex(TypeError, "'ranges' argument must be an iterable of NumericRange objects"):
            RangeSet(ranges=(NR(1, 5, 1), NNR('a')))
        with self.assertRaisesRegex(ValueError, 'Constructing a finite RangeSet over a non-finite range '):
            RangeSet(finite=True, ranges=(NR(1, 5, 0),))
        with self.assertRaisesRegex(ValueError, 'RangeSet does not support unbounded ranges with a non-integer step'):
            RangeSet(0, None, 0.5)
        with LoggingIntercept() as LOG:
            m = ConcreteModel()
            m.p = Param(initialize=5, mutable=False)
            m.I = RangeSet(0, m.p)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(RangeSet(0, 5, 1), m.I)
        with LoggingIntercept() as LOG:
            m = ConcreteModel()
            m.p = Param(initialize=5, mutable=True)
            m.I = RangeSet(0, m.p)
        self.assertIn("Constructing RangeSet 'I' from non-constant data", LOG.getvalue())
        self.assertEqual(RangeSet(0, 5, 1), m.I)

        class _AlmostNumeric(object):

            def __init__(self, val):
                self.val = val

            def __float__(self):
                return self.val

            def __add__(self, other):
                return self.val + other

            def __sub__(self, other):
                return self.val - other

            def __lt__(self, other):
                return self.val < other

            def __ge__(self, other):
                return self.val >= other
        i = RangeSet(_AlmostNumeric(1))
        self.assertFalse(i.is_constructed())
        i.construct()
        self.assertEqual(list(i), [1])
        output = StringIO()
        p = Param(initialize=5)
        i = RangeSet(p)
        self.assertFalse(i.is_constructed())
        self.assertIs(type(i), AbstractFiniteScalarRangeSet)
        p.construct()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertEqual(output.getvalue(), '')
            i.construct()
            ref = 'Constructing RangeSet, name=FiniteScalarRangeSet, from data=None\n'
            self.assertEqual(output.getvalue(), ref)
            self.assertTrue(i.is_constructed())
            self.assertIs(type(i), FiniteScalarRangeSet)
            i.construct()
            self.assertEqual(output.getvalue(), ref)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            i = SetOf([1, 2, 3])
            ref = 'Constructing SetOf, name=[1, 2, 3], from data=None\n'
            self.assertEqual(output.getvalue(), ref)
            i.construct()
            self.assertEqual(output.getvalue(), ref)
        i = RangeSet(0)
        self.assertEqual(len(i), 0)
        self.assertEqual(len(list(i.ranges())), 0)
        i = RangeSet(0, -1)
        self.assertEqual(len(i), 0)
        self.assertEqual(len(list(i.ranges())), 0)
        i = RangeSet(1, 10)
        self.assertIs(type(i), FiniteScalarRangeSet)
        i = RangeSet(1, 10, 0)
        self.assertIs(type(i), InfiniteScalarRangeSet)
        i = RangeSet(1, 1, 0)
        self.assertIs(type(i), FiniteScalarRangeSet)
        j = RangeSet(1, float('inf'))
        self.assertIs(type(j), InfiniteScalarRangeSet)
        i = RangeSet(1, None)
        self.assertIs(type(i), InfiniteScalarRangeSet)
        self.assertEqual(i, j)
        self.assertIn(1, i)
        self.assertIn(100, i)
        self.assertNotIn(0, i)
        self.assertNotIn(1.5, i)
        i = RangeSet(None, 1)
        self.assertIs(type(i), InfiniteScalarRangeSet)
        self.assertIn(1, i)
        self.assertNotIn(100, i)
        self.assertIn(0, i)
        self.assertNotIn(0.5, i)
        i = RangeSet(None, None)
        self.assertIs(type(i), InfiniteScalarRangeSet)
        self.assertIn(1, i)
        self.assertIn(100, i)
        self.assertIn(0, i)
        self.assertNotIn(0.5, i)
        i = RangeSet(None, None, bounds=(-5, 10))
        self.assertIs(type(i), InfiniteScalarRangeSet)
        self.assertIn(10, i)
        self.assertNotIn(11, i)
        self.assertIn(-5, i)
        self.assertNotIn(-6, i)
        self.assertNotIn(0.5, i)
        p = Param(initialize=float('inf'))
        i = RangeSet(1, p, 1)
        self.assertIs(type(i), AbstractFiniteScalarRangeSet)
        p.construct()
        i = RangeSet(1, p, 1)
        self.assertIs(type(i), InfiniteScalarRangeSet)
        m = AbstractModel()
        m.p = Param()
        m.q = Param()
        m.s = Param()
        m.i = RangeSet(m.p, m.q, m.s, finite=True)
        self.assertIs(type(m.i), AbstractFiniteScalarRangeSet)
        i = m.create_instance(data={None: {'p': {None: 1}, 'q': {None: 5}, 's': {None: 2}}})
        self.assertIs(type(i.i), FiniteScalarRangeSet)
        self.assertEqual(list(i.i), [1, 3, 5])
        with self.assertRaisesRegex(ValueError, 'finite RangeSet over a non-finite range \\(\\[1..5\\]\\)'):
            i = m.create_instance(data={None: {'p': {None: 1}, 'q': {None: 5}, 's': {None: 0}}})
        with self.assertRaisesRegex(ValueError, 'RangeSet.construct\\(\\) does not support the data= argument.'):
            i = m.create_instance(data={None: {'p': {None: 1}, 'q': {None: 5}, 's': {None: 1}, 'i': {None: [1, 2, 3]}}})

    def test_filter(self):

        def rFilter(m, i):
            return i % 2
        r = RangeSet(10, filter=rFilter)
        self.assertEqual(r, [1, 3, 5, 7, 9])
        r = RangeSet(1, filter=rFilter)
        self.assertEqual(r, [1])
        r = RangeSet(2, 2, filter=rFilter)
        self.assertEqual(r, [])
        r = RangeSet(2, 3, filter=rFilter)
        self.assertEqual(r, [3])

        def rFilter(m, i):
            return i is None or i % 2
        r = RangeSet(10, filter=rFilter)
        self.assertEqual(r, [1, 3, 5, 7, 9])
        with self.assertRaisesRegex(ValueError, "The 'filter' keyword argument is not valid for non-finite RangeSet component"):
            r = RangeSet(1, 10, 0, filter=rFilter)

    def test_validate(self):

        def rFilter(m, i):
            self.assertIs(m, None)
            return i % 2
        r = RangeSet(1, 10, 2, validate=rFilter)
        self.assertEqual(r, [1, 3, 5, 7, 9])
        with self.assertRaisesRegex(ValueError, 'The value=2 violates the validation rule'):
            r = RangeSet(10, validate=rFilter)

        def rFilter(m, i):
            return i is None or i % 2
        r = RangeSet(1, 10, 2, validate=rFilter)
        self.assertEqual(r, [1, 3, 5, 7, 9])
        with self.assertRaisesRegex(ValueError, "The 'validate' keyword argument is not valid for non-finite RangeSet component"):
            r = RangeSet(1, 10, 0, validate=rFilter)

        def badRule(m, i):
            raise RuntimeError('ERROR: %s' % i)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            with self.assertRaisesRegex(RuntimeError, 'ERROR: 1'):
                r = RangeSet(10, validate=badRule)
        self.assertEqual(output.getvalue(), "Exception raised while validating element '1' for Set FiniteScalarRangeSet\n")

    def test_bounds(self):
        r = RangeSet(100, bounds=(2.5, 5.5))
        self.assertEqual(r, [3, 4, 5])

    def test_contains(self):
        r = RangeSet(5)
        self.assertIn(1, r)
        self.assertIn((1,), r)
        self.assertNotIn(6, r)
        self.assertNotIn((6,), r)
        r = SetOf([1, (2,)])
        self.assertIn(1, r)
        self.assertIn((1,), r)
        self.assertNotIn(2, r)
        self.assertIn((2,), r)

    def test_equality(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.NotI = RangeSet(4)
        m.J = SetOf([1, 2, 3])
        m.NotJ = SetOf([1, 2, 3, 4])
        self.assertEqual(m.I, m.I)
        self.assertEqual(m.J, m.J)
        self.assertEqual(m.I, m.J)
        self.assertEqual(m.J, m.I)
        self.assertEqual(SetOf([1, 3, 4, 2]), SetOf({1, 2, 3, 4}))
        self.assertEqual(SetOf({1, 2, 3, 4}), SetOf([1, 3, 4, 2]))
        self.assertNotEqual(m.I, m.NotI)
        self.assertNotEqual(m.NotI, m.I)
        self.assertNotEqual(m.I, m.NotJ)
        self.assertNotEqual(m.NotJ, m.I)
        self.assertNotEqual(m.J, m.NotJ)
        self.assertNotEqual(m.NotJ, m.J)
        self.assertNotEqual(m.I, RangeSet(1, 3, 0))
        self.assertNotEqual(RangeSet(1, 3, 0), m.I)
        self.assertNotEqual(SetOf([1, 3, 5, 2]), SetOf({1, 2, 3, 4}))
        self.assertNotEqual(SetOf({1, 2, 3, 4}), SetOf([1, 3, 5, 2]))
        self.assertEqual(RangeSet(0, 4, 1), [0, 1, 2, 3, 4])
        self.assertEqual(RangeSet(0, 4), [0, 1, 2, 3, 4])
        self.assertEqual(RangeSet(4), [1, 2, 3, 4])

        class _NonIterable(object):

            def __init__(self):
                self.data = set({1, 3, 5})

            def __contains__(self, val):
                return val in self.data

            def __len__(self):
                return len(self.data)
        self.assertEqual(SetOf({1, 3, 5}), _NonIterable())
        self.assertNotEqual(SetOf({3}), 3)
        self.assertEqual(RangeSet(0.0, 2.0), RangeSet(0.0, 2.0))
        self.assertEqual(RangeSet(0.0, 2.0), RangeSet(0, 2))

    def test_inequality(self):
        self.assertTrue(SetOf([1, 2, 3]) <= SetOf({1, 2, 3}))
        self.assertFalse(SetOf([1, 2, 3]) < SetOf({1, 2, 3}))
        self.assertTrue(SetOf([1, 2, 3]) <= SetOf({1, 2, 3, 4}))
        self.assertTrue(SetOf([1, 2, 3]) < SetOf({1, 2, 3, 4}))
        self.assertFalse(SetOf([1, 2, 3]) <= SetOf({1, 2}))
        self.assertFalse(SetOf([1, 2, 3]) < SetOf({1, 2}))
        self.assertTrue(SetOf([1, 2, 3]) >= SetOf({1, 2, 3}))
        self.assertFalse(SetOf([1, 2, 3]) > SetOf({1, 2, 3}))
        self.assertFalse(SetOf([1, 2, 3]) >= SetOf({1, 2, 3, 4}))
        self.assertFalse(SetOf([1, 2, 3]) > SetOf({1, 2, 3, 4}))
        self.assertTrue(SetOf([1, 2, 3]) >= SetOf({1, 2}))
        self.assertTrue(SetOf([1, 2, 3]) > SetOf({1, 2}))

    def test_is_functions(self):
        i = SetOf({1, 2, 3})
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertFalse(i.isordered())
        i = SetOf([1, 2, 3])
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        i = SetOf((1, 2, 3))
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        i = RangeSet(3)
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        self.assertIsInstance(i, _FiniteRangeSetData)
        i = RangeSet(1, 3)
        self.assertTrue(i.isdiscrete())
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        self.assertIsInstance(i, _FiniteRangeSetData)
        i = RangeSet(1, 3, 0)
        self.assertFalse(i.isdiscrete())
        self.assertFalse(i.isfinite())
        self.assertFalse(i.isordered())
        self.assertIsInstance(i, _InfiniteRangeSetData)

    def test_pprint(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.K1 = RangeSet(0)
        m.K2 = RangeSet(10, 9)
        m.NotI = RangeSet(1, 3, 0)
        m.J = SetOf([1, 2, 3])
        buf = StringIO()
        m.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), '\n4 RangeSet Declarations\n    I : Dimen=1, Size=3, Bounds=(1, 3)\n        Key  : Finite : Members\n        None :   True :   [1:3]\n    K1 : Dimen=1, Size=0, Bounds=(None, None)\n        Key  : Finite : Members\n        None :   True :      []\n    K2 : Dimen=1, Size=0, Bounds=(None, None)\n        Key  : Finite : Members\n        None :   True :      []\n    NotI : Dimen=1, Size=Inf, Bounds=(1, 3)\n        Key  : Finite : Members\n        None :  False :  [1..3]\n\n1 SetOf Declarations\n    J : Dimen=1, Size=3, Bounds=(1, 3)\n        Key  : Ordered : Members\n        None :    True : [1, 2, 3]\n\n5 Declarations: I K1 K2 NotI J'.strip())

    def test_naming(self):
        m = ConcreteModel()
        i = RangeSet(3)
        self.assertEqual(str(i), '[1:3]')
        m.I = i
        self.assertEqual(str(i), 'I')
        j = RangeSet(ranges=(NR(1, 3, 0), NR(4, 7, 1)))
        self.assertEqual(str(j), '([1..3] | [4:7])')
        m.J = j
        self.assertEqual(str(j), 'J')
        k = SetOf((1, 3, 5))
        self.assertEqual(str(k), '(1, 3, 5)')
        m.K = k
        self.assertEqual(str(k), 'K')
        l = SetOf([1, 3, 5])
        self.assertEqual(str(l), '[1, 3, 5]')
        m.L = l
        self.assertEqual(str(l), 'L')
        n = RangeSet(0)
        self.assertEqual(str(n), '[]')
        m.N = n
        self.assertEqual(str(n), 'N')
        m.a = Param(initialize=3)
        o = RangeSet(m.a)
        self.assertEqual(str(o), '[1:3]')
        m.O = o
        self.assertEqual(str(o), 'O')
        p = RangeSet(m.a, finite=False)
        self.assertEqual(str(p), '[1:3]')
        m.P = p
        self.assertEqual(str(p), 'P')
        b = Param(initialize=3)
        oo = RangeSet(b)
        self.assertEqual(str(oo), 'AbstractFiniteScalarRangeSet')
        pp = RangeSet(b, finite=False)
        self.assertEqual(str(pp), 'AbstractInfiniteScalarRangeSet')
        b.construct()
        m.OO = oo
        self.assertEqual(str(oo), 'OO')
        m.PP = pp
        self.assertEqual(str(pp), 'PP')

    def test_isdisjoint(self):
        i = SetOf({1, 2, 3})
        self.assertTrue(i.isdisjoint({4, 5, 6}))
        self.assertFalse(i.isdisjoint({3, 4, 5, 6}))
        self.assertTrue(i.isdisjoint(SetOf({4, 5, 6})))
        self.assertFalse(i.isdisjoint(SetOf({3, 4, 5, 6})))
        self.assertTrue(i.isdisjoint(RangeSet(4, 6, 0)))
        self.assertFalse(i.isdisjoint(RangeSet(3, 6, 0)))
        self.assertTrue(RangeSet(4, 6, 0).isdisjoint(i))
        self.assertFalse(RangeSet(3, 6, 0).isdisjoint(i))
        _NonHashable = (1, 3, 5, [2, 3])
        self.assertFalse(SetOf([[2, 3], 4]).isdisjoint(_NonHashable))
        self.assertTrue(SetOf({0, 4}).isdisjoint(_NonHashable))
        self.assertFalse(SetOf(_NonHashable).isdisjoint(_NonHashable))
        self.assertFalse(SetOf((1, 3, 5)).isdisjoint(_NonHashable))

        class _NonIterable(object):

            def __init__(self):
                self.data = set({1, 3, 5})

            def __contains__(self, val):
                return val in self.data

            def __len__(self):
                return len(self.data)
        self.assertTrue(SetOf({2, 4}).isdisjoint(_NonIterable()))
        self.assertFalse(SetOf({2, 3, 4}).isdisjoint(_NonIterable()))
        with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
            i.isdisjoint(1)

    def test_issubset(self):
        i = SetOf({1, 2, 3})
        self.assertTrue(i.issubset({1, 2, 3, 4}))
        self.assertFalse(i.issubset({3, 4, 5, 6}))
        self.assertTrue(i.issubset(SetOf({1, 2, 3, 4})))
        self.assertFalse(i.issubset(SetOf({3, 4, 5, 6})))
        self.assertTrue(i.issubset(RangeSet(1, 4, 0)))
        self.assertFalse(i.issubset(RangeSet(3, 6, 0)))
        self.assertTrue(RangeSet(1, 3, 0).issubset(RangeSet(0, 100, 0)))
        self.assertFalse(RangeSet(1, 3, 0).issubset(i))
        self.assertFalse(RangeSet(3, 6, 0).issubset(i))
        _NonHashable = (1, 3, 5, [2, 3])
        self.assertFalse(SetOf({0, 1, 3, 5}).issubset(_NonHashable))
        self.assertTrue(SetOf({1, 3, 5}).issubset(_NonHashable))
        self.assertTrue(SetOf(_NonHashable).issubset(_NonHashable))
        self.assertTrue(SetOf((1, 3, 5, [2, 3])).issubset(_NonHashable))
        self.assertFalse(SetOf([2, 4]).issubset(_NonHashable))

        class _NonIterable(object):

            def __init__(self):
                self.data = set({1, 3, 5})

            def __contains__(self, val):
                return val in self.data

            def __len__(self):
                return len(self.data)
        self.assertTrue(SetOf({1, 5}).issubset(_NonIterable()))
        self.assertFalse(SetOf({1, 3, 4}).issubset(_NonIterable()))
        with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
            i.issubset(1)

    def test_issuperset(self):
        i = SetOf({1, 2, 3})
        self.assertTrue(i.issuperset({1, 2}))
        self.assertFalse(i.issuperset({3, 4, 5, 6}))
        self.assertTrue(i.issuperset(SetOf({1, 2})))
        self.assertFalse(i.issuperset(SetOf({3, 4, 5, 6})))
        self.assertFalse(i.issuperset(RangeSet(1, 3, 0)))
        self.assertFalse(i.issuperset(RangeSet(3, 6, 0)))
        self.assertTrue(RangeSet(1, 3, 0).issuperset(RangeSet(1, 2, 0)))
        self.assertTrue(RangeSet(1, 3, 0).issuperset(i))
        self.assertFalse(RangeSet(3, 6, 0).issuperset(i))
        _NonHashable = (1, 3, 5, [2, 3])
        self.assertFalse(SetOf({0, 1, 3, 5}).issuperset(_NonHashable))
        self.assertTrue(SetOf(([2, 3], 1, 2, 3, 5)).issuperset(_NonHashable))
        self.assertTrue(SetOf(_NonHashable).issuperset(_NonHashable))
        self.assertTrue(SetOf((1, 3, 5, [2, 3])).issuperset(_NonHashable))
        self.assertFalse(SetOf((1, 3, 5, [2, 4])).issuperset(_NonHashable))

        class _NonIterable(object):

            def __init__(self):
                self.data = set({1, 3, 5})

            def __contains__(self, val):
                return val in self.data

            def __len__(self):
                return len(self.data)
        with self.assertRaisesRegex(TypeError, 'not iterable'):
            SetOf({1, 5}).issuperset(_NonIterable())
        with self.assertRaisesRegex(TypeError, 'not iterable'):
            SetOf({1, 3, 4, 5}).issuperset(_NonIterable())
        with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
            i.issuperset(1)

    def test_unordered_setof(self):
        i = SetOf({1, 3, 2, 0})
        self.assertTrue(i.isfinite())
        self.assertFalse(i.isordered())
        self.assertEqual(i.ordered_data(), (0, 1, 2, 3))
        self.assertEqual(i.sorted_data(), (0, 1, 2, 3))
        self.assertEqual(tuple(reversed(i)), tuple(reversed(list(i))))

    def test_ordered_setof(self):
        i = SetOf([1, 3, 2, 0])
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        self.assertEqual(i.ordered_data(), (1, 3, 2, 0))
        self.assertEqual(i.sorted_data(), (0, 1, 2, 3))
        self.assertEqual(tuple(reversed(i)), (0, 2, 3, 1))
        self.assertEqual(i[2], 3)
        self.assertEqual(i[-1], 0)
        with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based: valid Set positional index values are \\[1 .. len\\(Set\\)\\] or \\[-1 .. -len\\(Set\\)\\]'):
            i[0]
        with self.assertRaisesRegex(IndexError, 'OrderedSetOf index out of range'):
            i[5]
        with self.assertRaisesRegex(IndexError, 'OrderedSetOf index out of range'):
            i[-5]
        self.assertEqual(i.ord(3), 2)
        with self.assertRaisesRegex(ValueError, '5 is not in list'):
            i.ord(5)
        self.assertEqual(i.first(), 1)
        self.assertEqual(i.last(), 0)
        self.assertEqual(i.next(3), 2)
        self.assertEqual(i.prev(2), 3)
        self.assertEqual(i.nextw(3), 2)
        self.assertEqual(i.prevw(2), 3)
        self.assertEqual(i.next(3, 2), 0)
        self.assertEqual(i.prev(2, 2), 1)
        self.assertEqual(i.nextw(3, 2), 0)
        self.assertEqual(i.prevw(2, 2), 1)
        with self.assertRaisesRegex(IndexError, 'Cannot advance past the end of the Set'):
            i.next(0)
        with self.assertRaisesRegex(IndexError, 'Cannot advance before the beginning of the Set'):
            i.prev(1)
        self.assertEqual(i.nextw(0), 1)
        self.assertEqual(i.prevw(1), 0)
        with self.assertRaisesRegex(IndexError, 'Cannot advance past the end of the Set'):
            i.next(0, 2)
        with self.assertRaisesRegex(IndexError, 'Cannot advance before the beginning of the Set'):
            i.prev(1, 2)
        self.assertEqual(i.nextw(0, 2), 3)
        self.assertEqual(i.prevw(1, 2), 2)
        i = SetOf((1, 3, 2, 0))
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        self.assertEqual(i.ordered_data(), (1, 3, 2, 0))
        self.assertEqual(i.sorted_data(), (0, 1, 2, 3))
        self.assertEqual(tuple(reversed(i)), (0, 2, 3, 1))
        self.assertEqual(i[2], 3)
        self.assertEqual(i[-1], 0)
        with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based: valid Set positional index values are \\[1 .. len\\(Set\\)\\] or \\[-1 .. -len\\(Set\\)\\]'):
            i[0]
        with self.assertRaisesRegex(IndexError, 'OrderedSetOf index out of range'):
            i[5]
        with self.assertRaisesRegex(IndexError, 'OrderedSetOf index out of range'):
            i[-5]
        self.assertEqual(i.ord(3), 2)
        with self.assertRaisesRegex(ValueError, 'x not in tuple'):
            i.ord(5)
        i = SetOf([1, None, 'a'])
        self.assertTrue(i.isfinite())
        self.assertTrue(i.isordered())
        self.assertEqual(i.ordered_data(), (1, None, 'a'))
        self.assertEqual(i.sorted_data(), (None, 1, 'a'))
        self.assertEqual(tuple(reversed(i)), ('a', None, 1))

    def test_ranges(self):
        i_data = [1, 3, 2, 0]
        i = SetOf(i_data)
        r = list(i.ranges())
        self.assertEqual(len(r), 4)
        for idx, x in enumerate(r):
            self.assertIsInstance(x, NR)
            self.assertTrue(x.isfinite())
            self.assertEqual(x.start, i[idx + 1])
            self.assertEqual(x.end, i[idx + 1])
            self.assertEqual(x.step, 0)
        try:
            self.assertIn(int, native_types)
            self.assertIn(int, native_numeric_types)
            native_types.remove(int)
            native_numeric_types.remove(int)
            r = list(i.ranges())
            self.assertEqual(len(r), 4)
            for idx, x in enumerate(r):
                self.assertIsInstance(x, NR)
                self.assertTrue(x.isfinite())
                self.assertEqual(x.start, i[idx + 1])
                self.assertEqual(x.end, i[idx + 1])
                self.assertEqual(x.step, 0)
            self.assertIn(int, native_types)
            self.assertIn(int, native_numeric_types)
        finally:
            native_types.add(int)
            native_numeric_types.add(int)
        i_data.append('abc')
        try:
            self.assertIn(str, native_types)
            self.assertNotIn(str, native_numeric_types)
            native_types.remove(str)
            r = list(i.ranges())
            self.assertEqual(len(r), 5)
            self.assertNotIn(str, native_types)
            self.assertNotIn(str, native_numeric_types)
            for idx, x in enumerate(r[:-1]):
                self.assertIsInstance(x, NR)
                self.assertTrue(x.isfinite())
                self.assertEqual(x.start, i[idx + 1])
                self.assertEqual(x.end, i[idx + 1])
                self.assertEqual(x.step, 0)
            self.assertIs(type(r[-1]), NNR)
        finally:
            native_types.add(str)

    def test_bounds(self):
        self.assertEqual(SetOf([1, 3, 2, 0]).bounds(), (0, 3))
        self.assertEqual(SetOf([1, 3.0, 2, 0]).bounds(), (0, 3.0))
        self.assertEqual(SetOf([None, 1, 'a']).bounds(), (None, None))
        self.assertEqual(SetOf(['apple', 'cat', 'bear']).bounds(), ('apple', 'cat'))
        self.assertEqual(RangeSet(ranges=(NR(0, 10, 2), NR(3, 20, 2))).bounds(), (0, 19))
        self.assertEqual(RangeSet(ranges=(NR(None, None, 0), NR(0, 10, 2))).bounds(), (None, None))
        self.assertEqual(RangeSet(ranges=(NR(100, None, -2), NR(0, 10, 2))).bounds(), (None, 100))
        self.assertEqual(RangeSet(ranges=(NR(-10, None, 2), NR(0, 10, 2))).bounds(), (-10, None))
        self.assertEqual(RangeSet(ranges=(NR(0, 10, 2), NR(None, None, 0))).bounds(), (None, None))
        self.assertEqual(RangeSet(ranges=(NR(0, 10, 2), NR(100, None, -2))).bounds(), (None, 100))
        self.assertEqual(RangeSet(ranges=(NR(0, 10, 2), NR(-10, None, 2))).bounds(), (-10, None))

    def test_dimen(self):
        self.assertEqual(SetOf([]).dimen, 0)
        self.assertEqual(SetOf([1, 2, 3]).dimen, 1)
        self.assertEqual(SetOf([(1, 2), (2, 3), (4, 5)]).dimen, 2)
        self.assertEqual(SetOf([1, (2, 3)]).dimen, None)
        self.assertEqual(SetOf(Integers).dimen, 1)
        self.assertEqual(SetOf(Binary).dimen, 1)
        m = ConcreteModel()
        m.I = Set(initialize=[(1, 2), (3, 4)])
        self.assertEqual(SetOf(m.I).dimen, 2)
        a = [1, 2, 3, 'abc']
        SetOf_a = SetOf(a)
        self.assertEqual(SetOf_a.dimen, 1)
        a.append((1, 2))
        self.assertEqual(SetOf_a.dimen, None)

    def test_rangeset_iter(self):
        i = RangeSet(0, 10, 2)
        self.assertEqual(tuple(i), (0, 2, 4, 6, 8, 10))
        self.assertEqual(tuple(i.ordered_iter()), (0, 2, 4, 6, 8, 10))
        self.assertEqual(tuple(i.sorted_iter()), (0, 2, 4, 6, 8, 10))
        i = RangeSet(ranges=(NR(0, 5, 2), NR(6, 10, 2)))
        self.assertEqual(tuple(i), (0, 2, 4, 6, 8, 10))
        i = RangeSet(ranges=(NR(0, 10, 2), NR(0, 10, 2)))
        self.assertEqual(tuple(i), (0, 2, 4, 6, 8, 10))
        i = RangeSet(ranges=(NR(0, 10, 2), NR(10, 0, -2)))
        self.assertEqual(tuple(i), (0, 2, 4, 6, 8, 10))
        i = RangeSet(ranges=(NR(0, 10, 2), NR(9, 0, -2)))
        self.assertEqual(tuple(i), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        i = RangeSet(ranges=(NR(0, 10, 2), NR(1, 10, 2)))
        self.assertEqual(tuple(i), tuple(range(11)))
        i = RangeSet(ranges=(NR(0, 30, 10), NR(12, 14, 1)))
        self.assertEqual(tuple(i), (0, 10, 12, 13, 14, 20, 30))
        i = RangeSet(ranges=(NR(0, 0, 0), NR(3, 3, 0), NR(2, 2, 0)))
        self.assertEqual(tuple(i), (0, 2, 3))

    def test_ord_index(self):
        r = RangeSet(2, 10, 2)
        for i, v in enumerate([2, 4, 6, 8, 10]):
            self.assertEqual(r.ord(v), i + 1)
            self.assertEqual(r[i + 1], v)
        with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based: valid Set positional index values are \\[1 .. len\\(Set\\)\\] or \\[-1 .. -len\\(Set\\)\\]'):
            r[0]
        with self.assertRaisesRegex(IndexError, 'FiniteScalarRangeSet index out of range'):
            r[10]
        with self.assertRaisesRegex(ValueError, 'Cannot identify position of 5 in Set'):
            r.ord(5)
        r = RangeSet(ranges=(NR(2, 10, 2), NR(6, 12, 3)))
        for i, v in enumerate([2, 4, 6, 8, 9, 10, 12]):
            self.assertEqual(r.ord(v), i + 1)
            self.assertEqual(r[i + 1], v)
        with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based: valid Set positional index values are \\[1 .. len\\(Set\\)\\] or \\[-1 .. -len\\(Set\\)\\]'):
            r[0]
        with self.assertRaisesRegex(IndexError, 'FiniteScalarRangeSet index out of range'):
            r[10]
        with self.assertRaisesRegex(ValueError, 'Cannot identify position of 5 in Set'):
            r.ord(5)
        so = SetOf([0, (1,), 1])
        self.assertEqual(so.ord((1,)), 2)
        self.assertEqual(so.ord(1), 3)

    def test_float_steps(self):
        a = RangeSet(0, 4, 0.5)
        self.assertEqual(len(a), 9)
        self.assertEqual(list(a - RangeSet(0, 4, 1)), [0.5, 1.5, 2.5, 3.5])
        with self.assertRaisesRegex(ValueError, 'RangeSet: start, end ordering incompatible with step direction \\(got \\[0:4:-0.5\\]\\)'):
            RangeSet(0, 4, -0.5)

    def test_check_values(self):
        m = ConcreteModel()
        m.I = RangeSet(5)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
            self.assertTrue(m.I.check_values())
        self.assertRegex(output.getvalue(), '^DEPRECATED: check_values\\(\\) is deprecated:')