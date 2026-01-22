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
class SimpleSetAordered(SimpleSetA):

    def setUp(self):
        PyomoModel.setUp(self)
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set A := 1 3 5 7; end;\n')
        OUTPUT.close()
        self.model.A = Set(ordered=True)
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

    def test_first(self):
        """Check that we can get the 'first' value in the set"""
        self.tmp = self.instance.A.first()
        self.assertNotEqual(self.tmp, None)
        self.assertEqual(self.tmp, 1)

    def test_ordered(self):
        tmp = []
        for val in self.instance.A:
            tmp.append(val)
        self.assertEqual(tmp, [1, 3, 5, 7])

    def test_getitem(self):
        self.assertEqual(self.instance.A[1], 1)
        self.assertEqual(self.instance.A[2], 3)
        self.assertEqual(self.instance.A[3], 5)
        self.assertEqual(self.instance.A[4], 7)
        self.assertEqual(self.instance.A[-1], 7)
        self.assertEqual(self.instance.A[-2], 5)
        self.assertEqual(self.instance.A[-3], 3)
        self.assertEqual(self.instance.A[-4], 1)
        self.assertRaises(IndexError, self.instance.A.__getitem__, 5)
        self.assertRaises(IndexError, self.instance.A.__getitem__, 0)
        self.assertRaises(IndexError, self.instance.A.__getitem__, -5)