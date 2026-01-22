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
class SimpleSetC(SimpleSetA):

    def setUp(self):
        PyomoModel.setUp(self)
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set A := (A1,1) (A3,1) (A5,1) (A7,1); end;\n')
        OUTPUT.close()
        self.model.A = Set(dimen=2)
        self.model.tmpset1 = Set(initialize=[('A1', 1), ('A3', 1), ('A5', 1), ('A7', 1)])
        self.model.tmpset2 = Set(initialize=[('A1', 1), ('A2', 1), ('A3', 1), ('A5', 1), ('A7', 1)])
        self.model.tmpset3 = Set(initialize=[('A2', 1), ('A3', 1), ('A5', 1), ('A7', 1), ('A9', 1)])
        self.model.setunion = Set(initialize=[('A1', 1), ('A2', 1), ('A3', 1), ('A5', 1), ('A7', 1), ('A9', 1)])
        self.model.setintersection = Set(initialize=[('A3', 1), ('A5', 1), ('A7', 1)])
        self.model.setxor = Set(initialize=[('A1', 1), ('A2', 1), ('A9', 1)])
        self.model.setdiff = Set(initialize=[('A1', 1)])
        self.model.setmul = Set(initialize=[('A1', 1, 'A2', 1), ('A1', 1, 'A3', 1), ('A1', 1, 'A5', 1), ('A1', 1, 'A7', 1), ('A1', 1, 'A9', 1), ('A3', 1, 'A2', 1), ('A3', 1, 'A3', 1), ('A3', 1, 'A5', 1), ('A3', 1, 'A7', 1), ('A3', 1, 'A9', 1), ('A5', 1, 'A2', 1), ('A5', 1, 'A3', 1), ('A5', 1, 'A5', 1), ('A5', 1, 'A7', 1), ('A5', 1, 'A9', 1), ('A7', 1, 'A2', 1), ('A7', 1, 'A3', 1), ('A7', 1, 'A5', 1), ('A7', 1, 'A7', 1), ('A7', 1, 'A9', 1)])
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.e1 = ('A1', 1)
        self.e2 = ('A2', 1)
        self.e3 = ('A3', 1)
        self.e4 = ('A4', 1)
        self.e5 = ('A5', 1)
        self.e6 = ('A6', 1)

    def tearDown(self):
        os.remove(currdir + 'setA.dat')
        PyomoModel.tearDown(self)

    def test_bounds(self):
        self.assertEqual(self.instance.A.bounds(), (('A1', 1), ('A7', 1)))

    def test_addInvalid(self):
        """Check that we get an error when adding invalid set elements"""
        self.assertEqual(self.instance.A.domain, Any)
        try:
            self.instance.A.add('2', '3', '4')
        except ValueError:
            pass
        else:
            self.fail('fail test_addInvalid')
        self.assertFalse('2' in self.instance.A, 'Found invalid new element in A')
        self.instance.A.add(('2', '3'))