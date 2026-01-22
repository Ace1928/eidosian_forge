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
class InfiniteSetTester(unittest.TestCase):

    def test_Reals(self):
        self.assertIn(0, Reals)
        self.assertIn(1.5, Reals)
        (self.assertIn(100, Reals),)
        (self.assertIn(-100, Reals),)
        self.assertNotIn('A', Reals)
        self.assertNotIn(None, Reals)
        self.assertFalse(Reals.isdiscrete())
        self.assertFalse(Reals.isfinite())
        self.assertEqual(Reals.dim(), 0)
        self.assertIs(Reals.index_set(), UnindexedComponent_set)
        with self.assertRaisesRegex(TypeError, ".*'GlobalSet' has no len"):
            len(Reals)
        with self.assertRaisesRegex(TypeError, "'GlobalSet' object is not iterable \\(non-finite Set 'Reals' is not iterable\\)"):
            list(Reals)
        self.assertEqual(list(Reals.ranges()), [NR(None, None, 0)])
        self.assertEqual(Reals.bounds(), (None, None))
        self.assertEqual(Reals.dimen, 1)
        tmp = RealSet()
        self.assertFalse(tmp.isdiscrete())
        self.assertFalse(tmp.isfinite())
        self.assertEqual(Reals, tmp)
        self.assertEqual(tmp, Reals)
        tmp.clear()
        self.assertEqual(EmptySet, tmp)
        self.assertEqual(tmp, EmptySet)
        self.assertEqual(tmp.domain, Reals)
        self.assertEqual(str(Reals), 'Reals')
        self.assertEqual(str(tmp), 'Reals')
        b = ConcreteModel()
        b.tmp = tmp
        self.assertEqual(str(tmp), 'tmp')

    def test_Integers(self):
        self.assertIn(0, Integers)
        self.assertNotIn(1.5, Integers)
        (self.assertIn(100, Integers),)
        (self.assertIn(-100, Integers),)
        self.assertNotIn('A', Integers)
        self.assertNotIn(None, Integers)
        self.assertTrue(Integers.isdiscrete())
        self.assertFalse(Integers.isfinite())
        self.assertEqual(Integers.dim(), 0)
        self.assertIs(Integers.index_set(), UnindexedComponent_set)
        with self.assertRaisesRegex(TypeError, ".*'GlobalSet' has no len"):
            len(Integers)
        with self.assertRaisesRegex(TypeError, "'GlobalSet' object is not iterable \\(non-finite Set 'Integers' is not iterable\\)"):
            list(Integers)
        self.assertEqual(list(Integers.ranges()), [NR(0, None, 1), NR(0, None, -1)])
        self.assertEqual(Integers.bounds(), (None, None))
        self.assertEqual(Integers.dimen, 1)
        tmp = IntegerSet()
        self.assertTrue(tmp.isdiscrete())
        self.assertFalse(tmp.isfinite())
        self.assertEqual(Integers, tmp)
        self.assertEqual(tmp, Integers)
        tmp.clear()
        self.assertEqual(EmptySet, tmp)
        self.assertEqual(tmp, EmptySet)
        self.assertEqual(tmp.domain, Reals)
        self.assertEqual(str(Integers), 'Integers')
        self.assertEqual(str(tmp), 'Integers')
        b = ConcreteModel()
        b.tmp = tmp
        self.assertEqual(str(tmp), 'tmp')

    def test_Any(self):
        self.assertIn(0, Any)
        self.assertIn(1.5, Any)
        (self.assertIn(100, Any),)
        (self.assertIn(-100, Any),)
        self.assertIn('A', Any)
        self.assertIn(None, Any)
        self.assertFalse(Any.isdiscrete())
        self.assertFalse(Any.isfinite())
        self.assertEqual(Any.dim(), 0)
        self.assertIs(Any.index_set(), UnindexedComponent_set)
        with self.assertRaisesRegex(TypeError, ".*'Any' has no len"):
            len(Any)
        with self.assertRaisesRegex(TypeError, "'GlobalSet' object is not iterable \\(non-finite Set 'Any' is not iterable\\)"):
            list(Any)
        self.assertEqual(list(Any.ranges()), [AnyRange()])
        self.assertEqual(Any.bounds(), (None, None))
        self.assertEqual(Any.dimen, None)
        tmp = _AnySet()
        self.assertFalse(tmp.isdiscrete())
        self.assertFalse(tmp.isfinite())
        self.assertEqual(Any, tmp)
        tmp.clear()
        self.assertEqual(Any, tmp)
        self.assertEqual(tmp.domain, Any)
        self.assertEqual(str(Any), 'Any')
        self.assertEqual(str(tmp), '_AnySet')
        b = ConcreteModel()
        b.tmp = tmp
        self.assertEqual(str(tmp), 'tmp')

    def test_AnyWithNone(self):
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            self.assertIn(None, AnyWithNone)
            self.assertIn(1, AnyWithNone)
        self.assertRegex(os.getvalue(), '^DEPRECATED: The AnyWithNone set is deprecated')
        self.assertEqual(Any, AnyWithNone)
        self.assertEqual(AnyWithNone, Any)

    def test_EmptySet(self):
        self.assertNotIn(0, EmptySet)
        self.assertNotIn(1.5, EmptySet)
        (self.assertNotIn(100, EmptySet),)
        (self.assertNotIn(-100, EmptySet),)
        self.assertNotIn('A', EmptySet)
        self.assertNotIn(None, EmptySet)
        self.assertTrue(EmptySet.isdiscrete())
        self.assertTrue(EmptySet.isfinite())
        self.assertEqual(EmptySet.dim(), 0)
        self.assertIs(EmptySet.index_set(), UnindexedComponent_set)
        self.assertEqual(len(EmptySet), 0)
        self.assertEqual(list(EmptySet), [])
        self.assertEqual(list(EmptySet.ranges()), [])
        self.assertEqual(EmptySet.bounds(), (None, None))
        self.assertEqual(EmptySet.dimen, 0)
        tmp = _EmptySet()
        self.assertTrue(tmp.isdiscrete())
        self.assertTrue(tmp.isfinite())
        self.assertEqual(EmptySet, tmp)
        tmp.clear()
        self.assertEqual(EmptySet, tmp)
        self.assertEqual(tmp.domain, EmptySet)
        self.assertEqual(str(EmptySet), 'EmptySet')
        self.assertEqual(str(tmp), '_EmptySet')
        b = ConcreteModel()
        b.tmp = tmp
        self.assertEqual(str(tmp), 'tmp')

    @unittest.skipIf(not numpy_available, 'NumPy required for these tests')
    def test_numpy_compatible(self):
        self.assertIn(np.intc(1), Reals)
        self.assertIn(np.float64(1), Reals)
        self.assertIn(np.float64(1.5), Reals)
        self.assertIn(np.intc(1), Integers)
        self.assertIn(np.float64(1), Integers)
        self.assertNotIn(np.float64(1.5), Integers)

    def test_relational_operators(self):
        Any2 = _AnySet()
        self.assertTrue(Any.issubset(Any2))
        self.assertTrue(Any.issuperset(Any2))
        self.assertFalse(Any.isdisjoint(Any2))
        Reals2 = RangeSet(ranges=(NR(None, None, 0),))
        self.assertTrue(Reals.issubset(Reals2))
        self.assertTrue(Reals.issuperset(Reals2))
        self.assertFalse(Reals.isdisjoint(Reals2))
        Integers2 = RangeSet(ranges=(NR(0, None, -1), NR(0, None, 1)))
        self.assertTrue(Integers.issubset(Integers2))
        self.assertTrue(Integers.issuperset(Integers2))
        self.assertFalse(Integers.isdisjoint(Integers2))
        self.assertTrue(Integers.issubset(Reals))
        self.assertFalse(Integers.issuperset(Reals))
        self.assertFalse(Integers.isdisjoint(Reals))
        self.assertFalse(Reals.issubset(Integers))
        self.assertTrue(Reals.issuperset(Integers))
        self.assertFalse(Reals.isdisjoint(Integers))
        self.assertTrue(Reals.issubset(Any))
        self.assertFalse(Reals.issuperset(Any))
        self.assertFalse(Reals.isdisjoint(Any))
        self.assertFalse(Any.issubset(Reals))
        self.assertTrue(Any.issuperset(Reals))
        self.assertFalse(Any.isdisjoint(Reals))
        self.assertFalse(Integers.issubset(PositiveIntegers))
        self.assertTrue(Integers.issuperset(PositiveIntegers))
        self.assertFalse(Integers.isdisjoint(PositiveIntegers))
        self.assertTrue(PositiveIntegers.issubset(Integers))
        self.assertFalse(PositiveIntegers.issuperset(Integers))
        self.assertFalse(PositiveIntegers.isdisjoint(Integers))
        tmp = IntegerSet()
        tmp.clear()
        self.assertTrue(tmp.issubset(EmptySet))
        self.assertTrue(tmp.issuperset(EmptySet))
        self.assertTrue(tmp.isdisjoint(EmptySet))
        self.assertTrue(EmptySet.issubset(tmp))
        self.assertTrue(EmptySet.issuperset(tmp))
        self.assertTrue(EmptySet.isdisjoint(tmp))

    def test_equality(self):
        self.assertEqual(Any, Any)
        self.assertEqual(Reals, Reals)
        self.assertEqual(PositiveIntegers, PositiveIntegers)
        self.assertEqual(Any, _AnySet())
        self.assertEqual(Reals, RangeSet(ranges=(NR(None, None, 0),)))
        self.assertEqual(Integers, RangeSet(ranges=(NR(0, None, -1), NR(0, None, 1))))
        self.assertNotEqual(Integers, Reals)
        self.assertNotEqual(Reals, Integers)
        self.assertNotEqual(Reals, Any)
        self.assertNotEqual(Any, Reals)
        self.assertEqual(RangeSet(ranges=(NR(0, None, -1), NR(0, None, 1))), RangeSet(ranges=(NR(0, None, 1), NR(0, None, -1))))
        self.assertEqual(RangeSet(ranges=(NR(10, None, -1), NR(10, None, 1))), RangeSet(ranges=(NR(0, None, 1), NR(0, None, -1))))
        self.assertEqual(RangeSet(ranges=(NR(0, None, -1), NR(0, None, 1))), RangeSet(ranges=(NR(10, None, 1), NR(10, None, -1))))
        self.assertEqual(PositiveIntegers, RangeSet(ranges=(NR(1, None, 2), NR(2, None, 2))))
        self.assertEqual(RangeSet(ranges=(NR(1, None, 2), NR(2, None, 2))), RangeSet(ranges=(NR(1, None, 3), NR(2, None, 3), NR(3, None, 3))))
        self.assertNotEqual(RangeSet(ranges=(NR(1, None, 2), NR(2, None, 2))), RangeSet(ranges=(NR(1, None, 3), NR(2, None, 3))))
        self.assertNotEqual(RangeSet(ranges=(NR(0, None, 2), NR(0, None, 2))), RangeSet(ranges=(NR(1, None, 3), NR(2, None, 3), NR(3, None, 3))))

    def test_bounds(self):
        self.assertEqual(Any.bounds(), (None, None))
        self.assertEqual(Reals.bounds(), (None, None))
        self.assertEqual(PositiveReals.bounds(), (0, None))
        self.assertEqual(NegativeIntegers.bounds(), (None, -1))