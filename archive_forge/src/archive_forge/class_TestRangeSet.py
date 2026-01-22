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
class TestRangeSet(SimpleSetA):

    def setUp(self):
        PyomoModel.setUp(self)
        self.model.A = RangeSet(1, 5)
        self.model.tmpset1 = Set(initialize=[1, 2, 3, 4, 5])
        self.model.tmpset2 = Set(initialize=[1, 2, 3, 4, 5, 7])
        self.model.tmpset3 = Set(initialize=[2, 3, 5, 7, 9])
        self.model.setunion = Set(initialize=[1, 2, 3, 4, 5, 7, 9])
        self.model.setintersection = Set(initialize=[2, 3, 5])
        self.model.setxor = Set(initialize=[1, 4, 7, 9])
        self.model.setdiff = Set(initialize=[1, 4])
        self.model.setmul = Set(initialize=[(1, 2), (1, 3), (1, 5), (1, 7), (1, 9), (2, 2), (2, 3), (2, 5), (2, 7), (2, 9), (3, 2), (3, 3), (3, 5), (3, 7), (3, 9), (4, 2), (4, 3), (4, 5), (4, 7), (4, 9), (5, 2), (5, 3), (5, 5), (5, 7), (5, 9)])
        self.instance = self.model.create_instance()
        self.e1 = 1
        self.e2 = 2
        self.e3 = 3
        self.e4 = 4
        self.e5 = 5
        self.e6 = 6

    def test_clear(self):
        """Check the clear() method empties the set"""
        self.instance.A.clear()
        self.assertEqual(len(self.instance.A), 0)

    def test_virtual(self):
        """Check if this is a virtual set"""
        self.assertEqual(self.instance.A.virtual, True)

    def test_ordered_getitem(self):
        """Check if this is a virtual set"""
        self.assertEqual(self.instance.A[1], 1)
        self.assertEqual(self.instance.A[2], 2)
        self.assertEqual(self.instance.A[3], 3)
        self.assertEqual(self.instance.A[4], 4)
        self.assertEqual(self.instance.A[5], 5)
        self.assertEqual(self.instance.A[-1], 5)
        self.assertEqual(self.instance.A[-2], 4)
        self.assertEqual(self.instance.A[-3], 3)
        self.assertEqual(self.instance.A[-4], 2)
        self.assertEqual(self.instance.A[-5], 1)
        self.assertRaises(IndexError, self.instance.A.__getitem__, 6)
        self.assertRaises(IndexError, self.instance.A.__getitem__, 0)
        self.assertRaises(IndexError, self.instance.A.__getitem__, -6)

    def test_bounds(self):
        """Verify the bounds on this set"""
        self.assertEqual(self.instance.A.bounds(), (1, 5))

    def test_addValid(self):
        """Check that we can add valid set elements"""
        with self.assertRaises(AttributeError):
            self.instance.A.add(6)

    def test_addInvalid(self):
        """Check that we get an error when adding invalid set elements"""
        with self.assertRaises(AttributeError):
            self.instance.A.add('2', '3', '4')
        self.assertFalse('2' in self.instance.A, 'Value we attempted to add is not in A')

    def test_removeValid(self):
        """Check that we can remove a valid set element"""
        with self.assertRaises(AttributeError):
            self.instance.A.remove(self.e3)
        self.assertEqual(len(self.instance.A), 5)
        self.assertTrue(self.e3 in self.instance.A, 'Element is still in A')

    def test_removeInvalid(self):
        """Check that we fail to remove an invalid set element"""
        with self.assertRaises(AttributeError):
            self.instance.A.remove(6)
        self.assertEqual(len(self.instance.A), 5)

    def test_remove(self):
        """Check that the elements are properly removed  by .remove"""
        pass

    def test_discardValid(self):
        """Check that we can discard a valid set element"""
        with self.assertRaises(AttributeError):
            self.instance.A.discard(self.e3)
        self.assertEqual(len(self.instance.A), 5)
        self.assertTrue(self.e3 in self.instance.A, 'Found element in A that attempted to discard')

    def test_discardInvalid(self):
        """Check that we fail to remove an invalid set element without an exception"""
        pass

    def test_contains(self):
        """Various checks for contains() method"""
        self.assertEqual(self.e1 in self.instance.A, True)
        self.assertEqual(self.e2 in self.instance.A, True)
        self.assertEqual('2' in self.instance.A, False)

    def test_len(self):
        """Check that a simple set of numeric elements has the right size"""
        self.assertEqual(len(self.instance.A), 5)

    def test_data(self):
        """Check that we can access the underlying set data"""
        self.assertEqual(len(self.instance.A.data()), 5)

    def test_filter_constructor(self):
        """Check that RangeSets can filter out unwanted elements"""

        def evenFilter(model, el):
            return el % 2 == 0
        self.instance.tmp = RangeSet(0, 10, filter=evenFilter)
        self.assertEqual(sorted([x for x in self.instance.tmp]), [0, 2, 4, 6, 8, 10])

    def test_filter_attribute(self):
        """Check that RangeSets can filter out unwanted elements"""

        def evenFilter(model, el):
            return el % 2 == 0
        self.instance.tmp = RangeSet(0, 10, filter=evenFilter)
        self.instance.tmp.construct()
        self.assertEqual(sorted([x for x in self.instance.tmp]), [0, 2, 4, 6, 8, 10])