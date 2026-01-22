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
class SimpleSetA(PyomoModel):

    def setUp(self):
        PyomoModel.setUp(self)
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set A := 1 3 5 7; end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.tmpset1 = Set(initialize=[1, 3, 5, 7])
        self.model.tmpset2 = Set(initialize=[1, 2, 3, 5, 7])
        self.model.tmpset3 = Set(initialize=[2, 3, 5, 7, 9])
        self.model.setunion = Set(initialize=[1, 2, 3, 5, 7, 9])
        self.model.setintersection = Set(initialize=[3, 5, 7])
        self.model.setxor = Set(initialize=[1, 2, 9])
        self.model.setdiff = Set(initialize=[1])
        self.model.setmul = Set(initialize=[(1, 2), (1, 3), (1, 5), (1, 7), (1, 9), (3, 2), (3, 3), (3, 5), (3, 7), (3, 9), (5, 2), (5, 3), (5, 5), (5, 7), (5, 9), (7, 2), (7, 3), (7, 5), (7, 7), (7, 9)])
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.e1 = 1
        self.e2 = 2
        self.e3 = 3
        self.e4 = 4
        self.e5 = 5
        self.e6 = 6

    def tearDown(self):
        if os.path.exists(currdir + 'setA.dat'):
            os.remove(currdir + 'setA.dat')
        PyomoModel.tearDown(self)

    def test_len(self):
        """Check that a simple set of numeric elements has the right size"""
        self.assertEqual(len(self.instance.A), 4)

    def test_data(self):
        """Check that we can access the underlying set data"""
        self.assertEqual(len(self.instance.A.data()), 4)

    def test_dim(self):
        """Check that a simple set has dimension zero for its indexing"""
        self.assertEqual(self.instance.A.dim(), 0)

    def test_clear(self):
        """Check the clear() method empties the set"""
        self.instance.A.clear()
        self.assertEqual(len(self.instance.A), 0)

    def test_virtual(self):
        """Check if this is not a virtual set"""
        self.assertEqual(self.instance.A.virtual, False)

    def test_bounds(self):
        """Verify the bounds on this set"""
        self.assertEqual(self.instance.A.bounds(), (1, 7))

    def test_check_values(self):
        """Check if the values added to this set are valid"""
        self.instance.A.check_values()

    def test_addValid(self):
        """Check that we can add valid set elements"""
        self.instance.A.add(self.e2, self.e4)
        self.assertEqual(len(self.instance.A), 6)
        self.assertFalse(self.e2 not in self.instance.A, 'Cannot find new element in A')
        self.assertFalse(self.e4 not in self.instance.A, 'Cannot find new element in A')

    def test_addInvalid(self):
        """Check that we get an error when adding invalid set elements"""
        self.assertEqual(self.instance.A.domain, Any)
        self.instance.A.add('2', '3', '4')
        self.assertFalse('2' not in self.instance.A, 'Found invalid new element in A')

    def test_removeValid(self):
        """Check that we can remove a valid set element"""
        self.instance.A.remove(self.e3)
        self.assertEqual(len(self.instance.A), 3)
        self.assertFalse(3 in self.instance.A, 'Found element in A that we removed')

    def test_removeInvalid(self):
        """Check that we fail to remove an invalid set element"""
        self.assertRaises(KeyError, self.instance.A.remove, 2)
        self.assertEqual(len(self.instance.A), 4)

    def test_discardValid(self):
        """Check that we can discard a valid set element"""
        self.instance.A.discard(self.e3)
        self.assertEqual(len(self.instance.A), 3)
        self.assertFalse(3 in self.instance.A, 'Found element in A that we removed')

    def test_discardInvalid(self):
        """Check that we fail to remove an invalid set element without an exception"""
        self.instance.A.discard(self.e2)
        self.assertEqual(len(self.instance.A), 4)

    def test_iterator(self):
        """Check that we can iterate through the set"""
        self.tmp = set()
        for val in self.instance.A:
            self.tmp.add(val)
        self.assertTrue(self.tmp == set(self.instance.A.data()), 'Set values found by the iterator appear to be different from the underlying set (%s) (%s)' % (str(self.tmp), str(self.instance.A.data())))

    def test_eq1(self):
        """Various checks for set equality and inequality (1)"""
        self.assertEqual(self.instance.A == self.instance.tmpset1, True)
        self.assertEqual(self.instance.tmpset1 == self.instance.A, True)
        self.assertEqual(self.instance.A != self.instance.tmpset1, False)
        self.assertEqual(self.instance.tmpset1 != self.instance.A, False)

    def test_eq2(self):
        """Various checks for set equality and inequality (2)"""
        self.assertEqual(self.instance.A == self.instance.tmpset2, False)
        self.assertEqual(self.instance.tmpset2 == self.instance.A, False)
        self.assertEqual(self.instance.A != self.instance.tmpset2, True)
        self.assertEqual(self.instance.tmpset2 != self.instance.A, True)

    def test_le1(self):
        """Various checks for set subset (1)"""
        self.assertEqual(self.instance.A < self.instance.tmpset1, False)
        self.assertEqual(self.instance.A <= self.instance.tmpset1, True)
        self.assertEqual(self.instance.A > self.instance.tmpset1, False)
        self.assertEqual(self.instance.A >= self.instance.tmpset1, True)
        self.assertEqual(self.instance.tmpset1 < self.instance.A, False)
        self.assertEqual(self.instance.tmpset1 <= self.instance.A, True)
        self.assertEqual(self.instance.tmpset1 > self.instance.A, False)
        self.assertEqual(self.instance.tmpset1 >= self.instance.A, True)

    def test_le2(self):
        """Various checks for set subset (2)"""
        self.assertEqual(self.instance.A < self.instance.tmpset2, True)
        self.assertEqual(self.instance.A <= self.instance.tmpset2, True)
        self.assertEqual(self.instance.A > self.instance.tmpset2, False)
        self.assertEqual(self.instance.A >= self.instance.tmpset2, False)
        self.assertEqual(self.instance.tmpset2 < self.instance.A, False)
        self.assertEqual(self.instance.tmpset2 <= self.instance.A, False)
        self.assertEqual(self.instance.tmpset2 > self.instance.A, True)
        self.assertEqual(self.instance.tmpset2 >= self.instance.A, True)

    def test_le3(self):
        """Various checks for set subset (3)"""
        self.assertEqual(self.instance.A < self.instance.tmpset3, False)
        self.assertEqual(self.instance.A <= self.instance.tmpset3, False)
        self.assertEqual(self.instance.A > self.instance.tmpset3, False)
        self.assertEqual(self.instance.A >= self.instance.tmpset3, False)
        self.assertEqual(self.instance.tmpset3 < self.instance.A, False)
        self.assertEqual(self.instance.tmpset3 <= self.instance.A, False)
        self.assertEqual(self.instance.tmpset3 > self.instance.A, False)
        self.assertEqual(self.instance.tmpset3 >= self.instance.A, False)

    def test_contains(self):
        """Various checks for contains() method"""
        self.assertEqual(self.e1 in self.instance.A, True)
        self.assertEqual(self.e2 in self.instance.A, False)
        self.assertEqual('2' in self.instance.A, False)

    def test_or(self):
        """Check that set union works"""
        self.instance.tmp = self.instance.A | self.instance.tmpset3
        self.instance.tmp.construct()
        self.assertEqual(self.instance.tmp == self.instance.setunion, True)

    def test_and(self):
        """Check that set intersection works"""
        self.instance.tmp = self.instance.A & self.instance.tmpset3
        self.instance.tmp.construct()
        self.assertEqual(self.instance.tmp == self.instance.setintersection, True)

    def test_xor(self):
        """Check that set exclusive or works"""
        self.instance.tmp = self.instance.A ^ self.instance.tmpset3
        self.instance.tmp.construct()
        self.assertEqual(self.instance.tmp == self.instance.setxor, True)

    def test_diff(self):
        """Check that set difference works"""
        self.instance.tmp = self.instance.A - self.instance.tmpset3
        self.instance.tmp.construct()
        self.assertEqual(self.instance.tmp == self.instance.setdiff, True)

    def test_mul(self):
        """Check that set cross-product works"""
        self.instance.tmp = self.instance.A * self.instance.tmpset3
        self.instance.tmp.construct()
        self.assertEqual(self.instance.tmp == self.instance.setmul, True)

    def test_filter_constructor(self):
        """Check that sets can filter out unwanted elements"""

        def evenFilter(model, el):
            return el % 2 == 0
        self.instance.tmp = Set(initialize=range(0, 10), filter=evenFilter)
        self.assertEqual(sorted([x for x in self.instance.tmp]), [0, 2, 4, 6, 8])

    def test_filter_attribute(self):
        """Check that sets can filter out unwanted elements"""

        def evenFilter(model, el):
            return el % 2 == 0
        m = AbstractModel()
        m.tmp = Set(initialize=range(0, 10), filter=evenFilter)
        m.tmp.construct()
        self.assertEqual(sorted([x for x in m.tmp]), [0, 2, 4, 6, 8])