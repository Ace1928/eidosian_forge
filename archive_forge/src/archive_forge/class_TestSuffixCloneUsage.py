import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
class TestSuffixCloneUsage(unittest.TestCase):

    def test_clone_VarElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x), None)
        model.junk.set_value(model.x, 1.0)
        self.assertEqual(model.junk.get(model.x), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x), None)
        self.assertEqual(inst.junk.get(inst.x), 1.0)

    def test_clone_VarArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.x[1]), None)
        model.junk.set_value(model.x, 1.0)
        self.assertEqual(model.junk.get(model.x), None)
        self.assertEqual(model.junk.get(model.x[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x[1]), None)
        self.assertEqual(inst.junk.get(inst.x[1]), 1.0)

    def test_clone_VarData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.x[1]), None)
        model.junk.set_value(model.x[1], 1.0)
        self.assertEqual(model.junk.get(model.x[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.x[1]), None)
        self.assertEqual(inst.junk.get(inst.x[1]), 1.0)

    def test_clone_ConstraintElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c), None)
        model.junk.set_value(model.c, 1.0)
        self.assertEqual(model.junk.get(model.c), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c), None)
        self.assertEqual(inst.junk.get(inst.c), 1.0)

    def test_clone_ConstraintArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.c = Constraint([1, 2, 3], rule=lambda model, i: model.x[i] == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c), None)
        self.assertEqual(model.junk.get(model.c[1]), None)
        model.junk.set_value(model.c, 1.0)
        self.assertEqual(model.junk.get(model.c), None)
        self.assertEqual(model.junk.get(model.c[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c[1]), None)
        self.assertEqual(inst.junk.get(inst.c[1]), 1.0)

    def test_clone_ConstraintData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.c = Constraint([1, 2, 3], rule=lambda model, i: model.x[i] == 1.0)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.c[1]), None)
        model.junk.set_value(model.c[1], 1.0)
        self.assertEqual(model.junk.get(model.c[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.c[1]), None)
        self.assertEqual(inst.junk.get(inst.c[1]), 1.0)

    def test_clone_ObjectiveElement(self):
        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj), None)
        model.junk.set_value(model.obj, 1.0)
        self.assertEqual(model.junk.get(model.obj), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj), None)
        self.assertEqual(inst.junk.get(inst.obj), 1.0)

    def test_clone_ObjectiveArray(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.obj = Objective([1, 2, 3], rule=lambda model, i: model.x[i])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj), None)
        self.assertEqual(model.junk.get(model.obj[1]), None)
        model.junk.set_value(model.obj, 1.0)
        self.assertEqual(model.junk.get(model.obj), None)
        self.assertEqual(model.junk.get(model.obj[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj[1]), None)
        self.assertEqual(inst.junk.get(inst.obj[1]), 1.0)

    def test_clone_ObjectiveData(self):
        model = ConcreteModel()
        model.x = Var([1, 2, 3], dense=True)
        model.obj = Objective([1, 2, 3], rule=lambda model, i: model.x[i])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.obj[1]), None)
        model.junk.set_value(model.obj[1], 1.0)
        self.assertEqual(model.junk.get(model.obj[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.obj[1]), None)
        self.assertEqual(inst.junk.get(inst.obj[1]), 1.0)

    def test_clone_SimpleBlock(self):
        model = ConcreteModel()
        model.b = Block()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b), None)
        model.junk.set_value(model.b, 1.0)
        self.assertEqual(model.junk.get(model.b), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b), None)
        self.assertEqual(inst.junk.get(inst.b), 1.0)

    def test_clone_IndexedBlock(self):
        model = ConcreteModel()
        model.b = Block([1, 2, 3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b), None)
        self.assertEqual(model.junk.get(model.b[1]), None)
        model.junk.set_value(model.b, 1.0)
        self.assertEqual(model.junk.get(model.b), None)
        self.assertEqual(model.junk.get(model.b[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b[1]), None)
        self.assertEqual(inst.junk.get(inst.b[1]), 1.0)

    def test_clone_BlockData(self):
        model = ConcreteModel()
        model.b = Block([1, 2, 3])
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model.b[1]), None)
        model.junk.set_value(model.b[1], 1.0)
        self.assertEqual(model.junk.get(model.b[1]), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model.b[1]), None)
        self.assertEqual(inst.junk.get(inst.b[1]), 1.0)

    def test_clone_model(self):
        model = ConcreteModel()
        model.junk = Suffix()
        self.assertEqual(model.junk.get(model), None)
        model.junk.set_value(model, 1.0)
        self.assertEqual(model.junk.get(model), 1.0)
        inst = model.clone()
        self.assertEqual(inst.junk.get(model), None)
        self.assertEqual(inst.junk.get(inst), 1.0)