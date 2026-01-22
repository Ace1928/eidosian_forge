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
class SimpleSetB(SimpleSetA):

    def setUp(self):
        PyomoModel.setUp(self)
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set A := A1 A3 A5 A7; end;\n')
        OUTPUT.close()
        self.model.A = Set()
        self.model.tmpset1 = Set(initialize=['A1', 'A3', 'A5', 'A7'])
        self.model.tmpset2 = Set(initialize=['A1', 'A2', 'A3', 'A5', 'A7'])
        self.model.tmpset3 = Set(initialize=['A2', 'A3', 'A5', 'A7', 'A9'])
        self.model.setunion = Set(initialize=['A1', 'A2', 'A3', 'A5', 'A7', 'A9'])
        self.model.setintersection = Set(initialize=['A3', 'A5', 'A7'])
        self.model.setxor = Set(initialize=['A1', 'A2', 'A9'])
        self.model.setdiff = Set(initialize=['A1'])
        self.model.setmul = Set(initialize=[('A1', 'A2'), ('A1', 'A3'), ('A1', 'A5'), ('A1', 'A7'), ('A1', 'A9'), ('A3', 'A2'), ('A3', 'A3'), ('A3', 'A5'), ('A3', 'A7'), ('A3', 'A9'), ('A5', 'A2'), ('A5', 'A3'), ('A5', 'A5'), ('A5', 'A7'), ('A5', 'A9'), ('A7', 'A2'), ('A7', 'A3'), ('A7', 'A5'), ('A7', 'A7'), ('A7', 'A9')])
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.e1 = 'A1'
        self.e2 = 'A2'
        self.e3 = 'A3'
        self.e4 = 'A4'
        self.e5 = 'A5'
        self.e6 = 'A6'

    def test_bounds(self):
        self.assertEqual(self.instance.A.bounds(), ('A1', 'A7'))