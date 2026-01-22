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
class TestSetArgs2(PyomoModel):

    def setUp(self):
        PyomoModel.setUp(self)

    def tearDown(self):
        if os.path.exists(currdir + 'setA.dat'):
            os.remove(currdir + 'setA.dat')
        PyomoModel.tearDown(self)

    def test_initialize(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set Z := A C; set A[A] := 1 3 5 7; end;')
        OUTPUT.close()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z, initialize={'A': [1, 2, 3, 'A']})
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.A['A']), 4)

    def test_dimen(self):
        self.model.Z = Set(initialize=[1, 2])
        self.model.A = Set(self.model.Z, initialize=[1, 2, 3], dimen=1)
        self.instance = self.model.create_instance()
        try:
            self.model.A = Set(self.model.Z, initialize=[4, 5, 6], dimen=2)
            self.instance = self.model.create_instance()
        except ValueError:
            pass
        else:
            self.fail('test_dimen')
        self.model.A = Set(self.model.Z, initialize=[(1, 2), (2, 3), (3, 4)], dimen=2)
        self.instance = self.model.create_instance()
        try:
            self.model.A = Set(self.model.Z, initialize=[(1, 2), (2, 3), (3, 4)], dimen=1)
            self.instance = self.model.create_instance()
        except ValueError:
            pass
        else:
            self.fail('test_dimen')

    def test_rule(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; param n := 5; set Z := A C; end;')
        OUTPUT.close()

        def tmp_init(model, i):
            return range(0, value(model.n))
        self.model.n = Param()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z, initialize=tmp_init)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.A['A']), 5)

    def test_rule2(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; param n := 5; set Z := A C; end;')
        OUTPUT.close()

        @simple_set_rule
        def tmp_rule2(model, z, i):
            if z > value(model.n):
                return None
            return z
        self.model.n = Param()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z, initialize=tmp_rule2)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.A['A']), 5)

    def test_rule3(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; param n := 5; set Z := A C; end;')
        OUTPUT.close()

        def tmp_rule2(model, z, i):
            if z > value(model.n):
                return Set.End
            return z
        self.model.n = Param()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z, initialize=tmp_rule2)
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.assertEqual(len(self.instance.A['A']), 5)

    def test_within1(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set Z := A C; set A[A] := 1 3 5 7.5; end;')
        OUTPUT.close()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z, within=Integers)
        try:
            self.instance = self.model.create_instance(currdir + 'setA.dat')
        except ValueError:
            pass
        else:
            self.fail('fail test_within1')

    def test_within2(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set Z := A C; set A[A] := 1 3 5 7.5; end;')
        OUTPUT.close()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z, within=Reals)
        try:
            self.instance = self.model.create_instance(currdir + 'setA.dat')
        except ValueError:
            self.fail('fail test_within2')
        else:
            pass

    def test_validation1(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set Z := A C; set A[A] := 1 3 5 7.5; end;')
        OUTPUT.close()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z, validate=lambda model, x: x < 6)
        try:
            self.instance = self.model.create_instance(currdir + 'setA.dat')
        except ValueError:
            pass
        else:
            self.fail('fail test_within1')

    def test_validation2(self):
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set Z := A C; set A[A] := 1 3 5 5.5; end;')
        OUTPUT.close()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z, validate=lambda model, x: x < 6)
        try:
            self.instance = self.model.create_instance(currdir + 'setA.dat')
        except ValueError:
            self.fail('fail test_within2')
        else:
            pass

    def test_other1(self):
        self.model.Z = Set(initialize=['A'])
        self.model.A = Set(self.model.Z, initialize={'A': [1, 2, 3, 'A']}, validate=lambda model, x: x in Integers)
        try:
            self.instance = self.model.create_instance()
        except ValueError:
            pass
        else:
            self.fail('fail test_other1')

    def test_other2(self):
        self.model.Z = Set(initialize=['A'])
        self.model.A = Set(self.model.Z, initialize={'A': [1, 2, 3, 'A']}, within=Integers)
        try:
            self.instance = self.model.create_instance()
        except ValueError:
            pass
        else:
            self.fail('fail test_other1')

    def test_other3(self):

        def tmp_init(model, i):
            tmp = []
            for i in range(0, value(model.n)):
                tmp.append(i / 2.0)
            return tmp
        self.model.n = Param(initialize=5)
        self.model.Z = Set(initialize=['A'])
        self.model.A = Set(self.model.Z, initialize=tmp_init, validate=lambda model, x: x in Integers)
        try:
            self.instance = self.model.create_instance()
        except ValueError:
            pass
        else:
            self.fail('fail test_other1')

    def test_other4(self):

        def tmp_init(model, i):
            tmp = []
            for i in range(0, value(model.n)):
                tmp.append(i / 2.0)
            return tmp
        self.model.n = Param(initialize=5)
        self.model.Z = Set(initialize=['A'])
        self.model.A = Set(self.model.Z, initialize=tmp_init, within=Integers)
        self.model.B = Set(self.model.Z, initialize=tmp_init, within=Integers)
        try:
            self.instance = self.model.create_instance()
        except ValueError:
            pass
        else:
            self.fail('fail test_other1')