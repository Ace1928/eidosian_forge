import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
class TestIntegerSet(unittest.TestCase):

    def test_bounds(self):
        x = IntegerSet()
        self.assertEqual(x.bounds(), (None, None))
        x = IntegerSet(bounds=(1, 2))
        self.assertEqual(x.bounds(), (1, 2))

    def test_inequality_comparison_fails(self):
        x = RealSet()
        y = RealSet()
        self.assertFalse(x < y)
        self.assertTrue(x <= y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)

    def test_name(self):
        x = IntegerSet()
        self.assertEqual(x.name, 'Integers')
        self.assertEqual('Integers', str(x))
        x = IntegerSet(name='x')
        self.assertEqual(x.name, 'x')
        self.assertEqual(str(x), 'x')

    def test_contains(self):
        x = IntegerSet()
        self.assertFalse(None in x)
        self.assertTrue(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertTrue(-10 in x)
        x = IntegerSet(bounds=(-1, 1))
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_PositiveIntegers(self):
        x = PositiveIntegers
        self.assertFalse(None in x)
        self.assertTrue(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertFalse(0.3 in x)
        self.assertFalse(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_NonPositiveIntegers(self):
        x = NonPositiveIntegers
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertFalse(1 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertTrue(-10 in x)

    def test_NegativeIntegers(self):
        x = NegativeIntegers
        self.assertFalse(None in x)
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertFalse(1 in x)
        self.assertFalse(0.3 in x)
        self.assertFalse(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertTrue(-10 in x)

    def test_NonNegativeIntegers(self):
        x = NonNegativeIntegers
        self.assertFalse(None in x)
        self.assertTrue(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertFalse(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)

    def test_IntegerInterval(self):
        x = IntegerInterval()
        self.assertFalse(None in x)
        self.assertEqual(x.name, "'IntegerInterval(None, None)'")
        self.assertEqual(x.local_name, 'IntegerInterval(None, None)')
        self.assertTrue(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertTrue(-10 in x)
        x = IntegerInterval(bounds=(-1, 1))
        self.assertFalse(None in x)
        self.assertEqual(x.name, "'IntegerInterval(-1, 1)'")
        self.assertEqual(x.local_name, 'IntegerInterval(-1, 1)')
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)
        x = IntegerInterval(bounds=(-1, 1), name='JUNK')
        self.assertFalse(None in x)
        self.assertEqual(x.name, 'JUNK')
        self.assertFalse(10 in x)
        self.assertFalse(1.1 in x)
        self.assertTrue(1 in x)
        self.assertFalse(0.3 in x)
        self.assertTrue(0 in x)
        self.assertFalse(-0.45 in x)
        self.assertTrue(-1 in x)
        self.assertFalse(-2.2 in x)
        self.assertFalse(-10 in x)